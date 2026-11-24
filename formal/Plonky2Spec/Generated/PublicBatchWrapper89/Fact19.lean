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

theorem publicBatchWrapper89_f14654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14518)) := by
  have c92430 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92431 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92432 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92433 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92434 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92435 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_6 := arithEq_of_rows h (row := 1507) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_6
  simp only [← c92430, ← c92431, ← c92432] at e_1507_6
  have e_1507_7 := arithEq_of_rows h (row := 1507) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_7
  simp only [← c92433, ← c92434, k2, ← c92435] at e_1507_7
  have hr := e_1507_7
  simp only [e_1507_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14519)) := by
  have c92436 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92437 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92438 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92439 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92440 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92441 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_8 := arithEq_of_rows h (row := 1507) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_8
  simp only [← c92436, ← c92437, ← c92438] at e_1507_8
  have e_1507_9 := arithEq_of_rows h (row := 1507) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_9
  simp only [← c92439, ← c92440, k2, ← c92441] at e_1507_9
  have hr := e_1507_9
  simp only [e_1507_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14520)) := by
  have c92442 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92443 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92444 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92445 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92446 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92447 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_10 := arithEq_of_rows h (row := 1507) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_10
  simp only [← c92442, ← c92443, ← c92444] at e_1507_10
  have e_1507_11 := arithEq_of_rows h (row := 1507) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_11
  simp only [← c92445, ← c92446, k2, ← c92447] at e_1507_11
  have hr := e_1507_11
  simp only [e_1507_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14521)) := by
  have c92448 := (publicBatchWrapper89_copies2889 a h).1
  have c92449 := (publicBatchWrapper89_copies2889 a h).2.1
  have c92450 := (publicBatchWrapper89_copies2889 a h).2.2.1
  have c92451 := (publicBatchWrapper89_copies2889 a h).2.2.2.1
  have c92452 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.1
  have c92453 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_12 := arithEq_of_rows h (row := 1507) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_12
  simp only [← c92448, ← c92449, ← c92450] at e_1507_12
  have e_1507_13 := arithEq_of_rows h (row := 1507) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_13
  simp only [← c92451, ← c92452, k2, ← c92453] at e_1507_13
  have hr := e_1507_13
  simp only [e_1507_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14522)) := by
  have c92454 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.1
  have c92455 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.1
  have c92456 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.1
  have c92457 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.1
  have c92458 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92459 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_14 := arithEq_of_rows h (row := 1507) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_14
  simp only [← c92454, ← c92455, ← c92456] at e_1507_14
  have e_1507_15 := arithEq_of_rows h (row := 1507) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_15
  simp only [← c92457, ← c92458, k2, ← c92459] at e_1507_15
  have hr := e_1507_15
  simp only [e_1507_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14523)) := by
  have c92460 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92461 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92462 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92463 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92464 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92465 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_16 := arithEq_of_rows h (row := 1507) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_16
  simp only [← c92460, ← c92461, ← c92462] at e_1507_16
  have e_1507_17 := arithEq_of_rows h (row := 1507) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_17
  simp only [← c92463, ← c92464, k2, ← c92465] at e_1507_17
  have hr := e_1507_17
  simp only [e_1507_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14524)) := by
  have c92466 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92467 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92468 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92469 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92470 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92471 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_18 := arithEq_of_rows h (row := 1507) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_18
  simp only [← c92466, ← c92467, ← c92468] at e_1507_18
  have e_1507_19 := arithEq_of_rows h (row := 1507) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_19
  simp only [← c92469, ← c92470, k2, ← c92471] at e_1507_19
  have hr := e_1507_19
  simp only [e_1507_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14525)) := by
  have c92472 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92473 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92474 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92475 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92476 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92477 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_0 := arithEq_of_rows h (row := 1508) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_0
  simp only [← c92472, ← c92473, ← c92474] at e_1508_0
  have e_1508_1 := arithEq_of_rows h (row := 1508) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_1
  simp only [← c92475, ← c92476, k2, ← c92477] at e_1508_1
  have hr := e_1508_1
  simp only [e_1508_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14526)) := by
  have c92478 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92479 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92480 := (publicBatchWrapper89_copies2890 a h).1
  have c92481 := (publicBatchWrapper89_copies2890 a h).2.1
  have c92482 := (publicBatchWrapper89_copies2890 a h).2.2.1
  have c92483 := (publicBatchWrapper89_copies2890 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_2 := arithEq_of_rows h (row := 1508) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_2
  simp only [← c92478, ← c92479, ← c92480] at e_1508_2
  have e_1508_3 := arithEq_of_rows h (row := 1508) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_3
  simp only [← c92481, ← c92482, k2, ← c92483] at e_1508_3
  have hr := e_1508_3
  simp only [e_1508_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14527)) := by
  have c92484 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.1
  have c92485 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.1
  have c92486 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.1
  have c92487 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.1
  have c92488 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.1
  have c92489 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_4 := arithEq_of_rows h (row := 1508) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_4
  simp only [← c92484, ← c92485, ← c92486] at e_1508_4
  have e_1508_5 := arithEq_of_rows h (row := 1508) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_5
  simp only [← c92487, ← c92488, k2, ← c92489] at e_1508_5
  have hr := e_1508_5
  simp only [e_1508_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14528)) := by
  have c92490 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92491 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92492 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92493 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92494 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92495 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_6 := arithEq_of_rows h (row := 1508) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_6
  simp only [← c92490, ← c92491, ← c92492] at e_1508_6
  have e_1508_7 := arithEq_of_rows h (row := 1508) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_7
  simp only [← c92493, ← c92494, k2, ← c92495] at e_1508_7
  have hr := e_1508_7
  simp only [e_1508_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14529)) := by
  have c92496 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92497 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92498 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92499 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92500 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92501 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_8 := arithEq_of_rows h (row := 1508) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_8
  simp only [← c92496, ← c92497, ← c92498] at e_1508_8
  have e_1508_9 := arithEq_of_rows h (row := 1508) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_9
  simp only [← c92499, ← c92500, k2, ← c92501] at e_1508_9
  have hr := e_1508_9
  simp only [e_1508_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14530)) := by
  have c92502 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92503 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92504 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92505 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92506 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92507 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_10 := arithEq_of_rows h (row := 1508) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_10
  simp only [← c92502, ← c92503, ← c92504] at e_1508_10
  have e_1508_11 := arithEq_of_rows h (row := 1508) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_11
  simp only [← c92505, ← c92506, k2, ← c92507] at e_1508_11
  have hr := e_1508_11
  simp only [e_1508_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14531)) := by
  have c92508 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92509 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92510 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92511 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92512 := (publicBatchWrapper89_copies2891 a h).1
  have c92513 := (publicBatchWrapper89_copies2891 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_12 := arithEq_of_rows h (row := 1508) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_12
  simp only [← c92508, ← c92509, ← c92510] at e_1508_12
  have e_1508_13 := arithEq_of_rows h (row := 1508) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_13
  simp only [← c92511, ← c92512, k2, ← c92513] at e_1508_13
  have hr := e_1508_13
  simp only [e_1508_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14532)) := by
  have c92514 := (publicBatchWrapper89_copies2891 a h).2.2.1
  have c92515 := (publicBatchWrapper89_copies2891 a h).2.2.2.1
  have c92516 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.1
  have c92517 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.1
  have c92518 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.1
  have c92519 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_14 := arithEq_of_rows h (row := 1508) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_14
  simp only [← c92514, ← c92515, ← c92516] at e_1508_14
  have e_1508_15 := arithEq_of_rows h (row := 1508) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_15
  simp only [← c92517, ← c92518, k2, ← c92519] at e_1508_15
  have hr := e_1508_15
  simp only [e_1508_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14533)) := by
  have c92520 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.1
  have c92521 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.1
  have c92522 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92523 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92524 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92525 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_16 := arithEq_of_rows h (row := 1508) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_16
  simp only [← c92520, ← c92521, ← c92522] at e_1508_16
  have e_1508_17 := arithEq_of_rows h (row := 1508) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_17
  simp only [← c92523, ← c92524, k2, ← c92525] at e_1508_17
  have hr := e_1508_17
  simp only [e_1508_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14534)) := by
  have c92526 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92527 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92528 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92529 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92530 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92531 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_18 := arithEq_of_rows h (row := 1508) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_18
  simp only [← c92526, ← c92527, ← c92528] at e_1508_18
  have e_1508_19 := arithEq_of_rows h (row := 1508) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_19
  simp only [← c92529, ← c92530, k2, ← c92531] at e_1508_19
  have hr := e_1508_19
  simp only [e_1508_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14535)) := by
  have c92532 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92533 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92534 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92535 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92536 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92537 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_0 := arithEq_of_rows h (row := 1509) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_0
  simp only [← c92532, ← c92533, ← c92534] at e_1509_0
  have e_1509_1 := arithEq_of_rows h (row := 1509) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_1
  simp only [← c92535, ← c92536, k2, ← c92537] at e_1509_1
  have hr := e_1509_1
  simp only [e_1509_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14536)) := by
  have c92538 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92539 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92540 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92541 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92542 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92543 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_2 := arithEq_of_rows h (row := 1509) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_2
  simp only [← c92538, ← c92539, ← c92540] at e_1509_2
  have e_1509_3 := arithEq_of_rows h (row := 1509) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_3
  simp only [← c92541, ← c92542, k2, ← c92543] at e_1509_3
  have hr := e_1509_3
  simp only [e_1509_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14537)) := by
  have c92544 := (publicBatchWrapper89_copies2892 a h).1
  have c92545 := (publicBatchWrapper89_copies2892 a h).2.1
  have c92546 := (publicBatchWrapper89_copies2892 a h).2.2.1
  have c92547 := (publicBatchWrapper89_copies2892 a h).2.2.2.1
  have c92548 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.1
  have c92549 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_4 := arithEq_of_rows h (row := 1509) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_4
  simp only [← c92544, ← c92545, ← c92546] at e_1509_4
  have e_1509_5 := arithEq_of_rows h (row := 1509) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_5
  simp only [← c92547, ← c92548, k2, ← c92549] at e_1509_5
  have hr := e_1509_5
  simp only [e_1509_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14538)) := by
  have c92550 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.1
  have c92551 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.1
  have c92552 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.1
  have c92553 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.1
  have c92554 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92555 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_6 := arithEq_of_rows h (row := 1509) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_6
  simp only [← c92550, ← c92551, ← c92552] at e_1509_6
  have e_1509_7 := arithEq_of_rows h (row := 1509) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_7
  simp only [← c92553, ← c92554, k2, ← c92555] at e_1509_7
  have hr := e_1509_7
  simp only [e_1509_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14539)) := by
  have c92556 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92557 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92558 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92559 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92560 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92561 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_8 := arithEq_of_rows h (row := 1509) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_8
  simp only [← c92556, ← c92557, ← c92558] at e_1509_8
  have e_1509_9 := arithEq_of_rows h (row := 1509) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_9
  simp only [← c92559, ← c92560, k2, ← c92561] at e_1509_9
  have hr := e_1509_9
  simp only [e_1509_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14540)) := by
  have c92562 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92563 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92564 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92565 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92566 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92567 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_10 := arithEq_of_rows h (row := 1509) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_10
  simp only [← c92562, ← c92563, ← c92564] at e_1509_10
  have e_1509_11 := arithEq_of_rows h (row := 1509) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_11
  simp only [← c92565, ← c92566, k2, ← c92567] at e_1509_11
  have hr := e_1509_11
  simp only [e_1509_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14541)) := by
  have c92568 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92569 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92570 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92571 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92572 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92573 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_12 := arithEq_of_rows h (row := 1509) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_12
  simp only [← c92568, ← c92569, ← c92570] at e_1509_12
  have e_1509_13 := arithEq_of_rows h (row := 1509) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_13
  simp only [← c92571, ← c92572, k2, ← c92573] at e_1509_13
  have hr := e_1509_13
  simp only [e_1509_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14542)) := by
  have c92574 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92575 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92576 := (publicBatchWrapper89_copies2893 a h).1
  have c92577 := (publicBatchWrapper89_copies2893 a h).2.1
  have c92578 := (publicBatchWrapper89_copies2893 a h).2.2.1
  have c92579 := (publicBatchWrapper89_copies2893 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_14 := arithEq_of_rows h (row := 1509) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_14
  simp only [← c92574, ← c92575, ← c92576] at e_1509_14
  have e_1509_15 := arithEq_of_rows h (row := 1509) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_15
  simp only [← c92577, ← c92578, k2, ← c92579] at e_1509_15
  have hr := e_1509_15
  simp only [e_1509_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14543)) := by
  have c92580 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.1
  have c92581 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.1
  have c92582 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.1
  have c92583 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.1
  have c92584 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.1
  have c92585 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_16 := arithEq_of_rows h (row := 1509) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_16
  simp only [← c92580, ← c92581, ← c92582] at e_1509_16
  have e_1509_17 := arithEq_of_rows h (row := 1509) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_17
  simp only [← c92583, ← c92584, k2, ← c92585] at e_1509_17
  have hr := e_1509_17
  simp only [e_1509_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14544)) := by
  have c92586 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92587 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92588 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92589 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92590 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92591 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_18 := arithEq_of_rows h (row := 1509) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_18
  simp only [← c92586, ← c92587, ← c92588] at e_1509_18
  have e_1509_19 := arithEq_of_rows h (row := 1509) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_19
  simp only [← c92589, ← c92590, k2, ← c92591] at e_1509_19
  have hr := e_1509_19
  simp only [e_1509_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14545)) := by
  have c92592 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92593 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92594 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92595 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92596 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92597 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_0 := arithEq_of_rows h (row := 1510) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_0
  simp only [← c92592, ← c92593, ← c92594] at e_1510_0
  have e_1510_1 := arithEq_of_rows h (row := 1510) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_1
  simp only [← c92595, ← c92596, k2, ← c92597] at e_1510_1
  have hr := e_1510_1
  simp only [e_1510_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14546)) := by
  have c92598 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92599 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92600 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92601 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92602 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92603 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_2 := arithEq_of_rows h (row := 1510) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_2
  simp only [← c92598, ← c92599, ← c92600] at e_1510_2
  have e_1510_3 := arithEq_of_rows h (row := 1510) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_3
  simp only [← c92601, ← c92602, k2, ← c92603] at e_1510_3
  have hr := e_1510_3
  simp only [e_1510_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14547)) := by
  have c92604 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92605 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92606 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92607 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92608 := (publicBatchWrapper89_copies2894 a h).1
  have c92609 := (publicBatchWrapper89_copies2894 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_4 := arithEq_of_rows h (row := 1510) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_4
  simp only [← c92604, ← c92605, ← c92606] at e_1510_4
  have e_1510_5 := arithEq_of_rows h (row := 1510) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_5
  simp only [← c92607, ← c92608, k2, ← c92609] at e_1510_5
  have hr := e_1510_5
  simp only [e_1510_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14548)) := by
  have c92610 := (publicBatchWrapper89_copies2894 a h).2.2.1
  have c92611 := (publicBatchWrapper89_copies2894 a h).2.2.2.1
  have c92612 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.1
  have c92613 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.1
  have c92614 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.1
  have c92615 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_6 := arithEq_of_rows h (row := 1510) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_6
  simp only [← c92610, ← c92611, ← c92612] at e_1510_6
  have e_1510_7 := arithEq_of_rows h (row := 1510) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_7
  simp only [← c92613, ← c92614, k2, ← c92615] at e_1510_7
  have hr := e_1510_7
  simp only [e_1510_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14549)) := by
  have c92616 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.1
  have c92617 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.1
  have c92618 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92619 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92620 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92621 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_8 := arithEq_of_rows h (row := 1510) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_8
  simp only [← c92616, ← c92617, ← c92618] at e_1510_8
  have e_1510_9 := arithEq_of_rows h (row := 1510) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_9
  simp only [← c92619, ← c92620, k2, ← c92621] at e_1510_9
  have hr := e_1510_9
  simp only [e_1510_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14550)) := by
  have c92622 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92623 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92624 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92625 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92626 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92627 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_10 := arithEq_of_rows h (row := 1510) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_10
  simp only [← c92622, ← c92623, ← c92624] at e_1510_10
  have e_1510_11 := arithEq_of_rows h (row := 1510) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_11
  simp only [← c92625, ← c92626, k2, ← c92627] at e_1510_11
  have hr := e_1510_11
  simp only [e_1510_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14551)) := by
  have c92628 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92629 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92630 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92631 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92632 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92633 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_12 := arithEq_of_rows h (row := 1510) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_12
  simp only [← c92628, ← c92629, ← c92630] at e_1510_12
  have e_1510_13 := arithEq_of_rows h (row := 1510) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_13
  simp only [← c92631, ← c92632, k2, ← c92633] at e_1510_13
  have hr := e_1510_13
  simp only [e_1510_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14552)) := by
  have c92634 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92635 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92636 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92637 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92638 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92639 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_14 := arithEq_of_rows h (row := 1510) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_14
  simp only [← c92634, ← c92635, ← c92636] at e_1510_14
  have e_1510_15 := arithEq_of_rows h (row := 1510) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_15
  simp only [← c92637, ← c92638, k2, ← c92639] at e_1510_15
  have hr := e_1510_15
  simp only [e_1510_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14553)) := by
  have c92640 := (publicBatchWrapper89_copies2895 a h).1
  have c92641 := (publicBatchWrapper89_copies2895 a h).2.1
  have c92642 := (publicBatchWrapper89_copies2895 a h).2.2.1
  have c92643 := (publicBatchWrapper89_copies2895 a h).2.2.2.1
  have c92644 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.1
  have c92645 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_16 := arithEq_of_rows h (row := 1510) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_16
  simp only [← c92640, ← c92641, ← c92642] at e_1510_16
  have e_1510_17 := arithEq_of_rows h (row := 1510) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_17
  simp only [← c92643, ← c92644, k2, ← c92645] at e_1510_17
  have hr := e_1510_17
  simp only [e_1510_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14554)) := by
  have c92646 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.1
  have c92647 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.1
  have c92648 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.1
  have c92649 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.1
  have c92650 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92651 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_18 := arithEq_of_rows h (row := 1510) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_18
  simp only [← c92646, ← c92647, ← c92648] at e_1510_18
  have e_1510_19 := arithEq_of_rows h (row := 1510) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_19
  simp only [← c92649, ← c92650, k2, ← c92651] at e_1510_19
  have hr := e_1510_19
  simp only [e_1510_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14555)) := by
  have c92652 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92653 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92654 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92655 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92656 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92657 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_0 := arithEq_of_rows h (row := 1511) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_0
  simp only [← c92652, ← c92653, ← c92654] at e_1511_0
  have e_1511_1 := arithEq_of_rows h (row := 1511) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_1
  simp only [← c92655, ← c92656, k2, ← c92657] at e_1511_1
  have hr := e_1511_1
  simp only [e_1511_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14556)) := by
  have c92658 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92659 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92660 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92661 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92662 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92663 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_2 := arithEq_of_rows h (row := 1511) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_2
  simp only [← c92658, ← c92659, ← c92660] at e_1511_2
  have e_1511_3 := arithEq_of_rows h (row := 1511) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_3
  simp only [← c92661, ← c92662, k2, ← c92663] at e_1511_3
  have hr := e_1511_3
  simp only [e_1511_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14557)) := by
  have c92664 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92665 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92666 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92667 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92668 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92669 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_4 := arithEq_of_rows h (row := 1511) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_4
  simp only [← c92664, ← c92665, ← c92666] at e_1511_4
  have e_1511_5 := arithEq_of_rows h (row := 1511) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_5
  simp only [← c92667, ← c92668, k2, ← c92669] at e_1511_5
  have hr := e_1511_5
  simp only [e_1511_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14558)) := by
  have c92670 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92671 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92672 := (publicBatchWrapper89_copies2896 a h).1
  have c92673 := (publicBatchWrapper89_copies2896 a h).2.1
  have c92674 := (publicBatchWrapper89_copies2896 a h).2.2.1
  have c92675 := (publicBatchWrapper89_copies2896 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_6 := arithEq_of_rows h (row := 1511) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_6
  simp only [← c92670, ← c92671, ← c92672] at e_1511_6
  have e_1511_7 := arithEq_of_rows h (row := 1511) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_7
  simp only [← c92673, ← c92674, k2, ← c92675] at e_1511_7
  have hr := e_1511_7
  simp only [e_1511_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14559)) := by
  have c92676 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.1
  have c92677 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.1
  have c92678 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.1
  have c92679 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.1
  have c92680 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.1
  have c92681 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_8 := arithEq_of_rows h (row := 1511) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_8
  simp only [← c92676, ← c92677, ← c92678] at e_1511_8
  have e_1511_9 := arithEq_of_rows h (row := 1511) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_9
  simp only [← c92679, ← c92680, k2, ← c92681] at e_1511_9
  have hr := e_1511_9
  simp only [e_1511_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14560)) := by
  have c92682 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92683 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92684 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92685 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92686 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92687 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_10 := arithEq_of_rows h (row := 1511) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_10
  simp only [← c92682, ← c92683, ← c92684] at e_1511_10
  have e_1511_11 := arithEq_of_rows h (row := 1511) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_11
  simp only [← c92685, ← c92686, k2, ← c92687] at e_1511_11
  have hr := e_1511_11
  simp only [e_1511_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14561)) := by
  have c92688 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92689 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92690 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92691 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92692 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92693 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_12 := arithEq_of_rows h (row := 1511) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_12
  simp only [← c92688, ← c92689, ← c92690] at e_1511_12
  have e_1511_13 := arithEq_of_rows h (row := 1511) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_13
  simp only [← c92691, ← c92692, k2, ← c92693] at e_1511_13
  have hr := e_1511_13
  simp only [e_1511_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14562)) := by
  have c92694 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92695 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92696 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92697 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92698 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92699 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_14 := arithEq_of_rows h (row := 1511) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_14
  simp only [← c92694, ← c92695, ← c92696] at e_1511_14
  have e_1511_15 := arithEq_of_rows h (row := 1511) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_15
  simp only [← c92697, ← c92698, k2, ← c92699] at e_1511_15
  have hr := e_1511_15
  simp only [e_1511_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14563)) := by
  have c92700 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92701 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92702 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92703 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92704 := (publicBatchWrapper89_copies2897 a h).1
  have c92705 := (publicBatchWrapper89_copies2897 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_16 := arithEq_of_rows h (row := 1511) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_16
  simp only [← c92700, ← c92701, ← c92702] at e_1511_16
  have e_1511_17 := arithEq_of_rows h (row := 1511) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_17
  simp only [← c92703, ← c92704, k2, ← c92705] at e_1511_17
  have hr := e_1511_17
  simp only [e_1511_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14564)) := by
  have c92706 := (publicBatchWrapper89_copies2897 a h).2.2.1
  have c92707 := (publicBatchWrapper89_copies2897 a h).2.2.2.1
  have c92708 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.1
  have c92709 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.1
  have c92710 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.1
  have c92711 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_18 := arithEq_of_rows h (row := 1511) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_18
  simp only [← c92706, ← c92707, ← c92708] at e_1511_18
  have e_1511_19 := arithEq_of_rows h (row := 1511) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_19
  simp only [← c92709, ← c92710, k2, ← c92711] at e_1511_19
  have hr := e_1511_19
  simp only [e_1511_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14565)) := by
  have c92712 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.1
  have c92713 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.1
  have c92714 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92715 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92716 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92717 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_0 := arithEq_of_rows h (row := 1512) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_0
  simp only [← c92712, ← c92713, ← c92714] at e_1512_0
  have e_1512_1 := arithEq_of_rows h (row := 1512) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_1
  simp only [← c92715, ← c92716, k2, ← c92717] at e_1512_1
  have hr := e_1512_1
  simp only [e_1512_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14566)) := by
  have c92718 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92719 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92720 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92721 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92722 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92723 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_2 := arithEq_of_rows h (row := 1512) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_2
  simp only [← c92718, ← c92719, ← c92720] at e_1512_2
  have e_1512_3 := arithEq_of_rows h (row := 1512) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_3
  simp only [← c92721, ← c92722, k2, ← c92723] at e_1512_3
  have hr := e_1512_3
  simp only [e_1512_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14567)) := by
  have c92724 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92725 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92726 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92727 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92728 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92729 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_4 := arithEq_of_rows h (row := 1512) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_4
  simp only [← c92724, ← c92725, ← c92726] at e_1512_4
  have e_1512_5 := arithEq_of_rows h (row := 1512) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_5
  simp only [← c92727, ← c92728, k2, ← c92729] at e_1512_5
  have hr := e_1512_5
  simp only [e_1512_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14568)) := by
  have c92730 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92731 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92732 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92733 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92734 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92735 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_6 := arithEq_of_rows h (row := 1512) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_6
  simp only [← c92730, ← c92731, ← c92732] at e_1512_6
  have e_1512_7 := arithEq_of_rows h (row := 1512) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_7
  simp only [← c92733, ← c92734, k2, ← c92735] at e_1512_7
  have hr := e_1512_7
  simp only [e_1512_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14569)) := by
  have c92736 := (publicBatchWrapper89_copies2898 a h).1
  have c92737 := (publicBatchWrapper89_copies2898 a h).2.1
  have c92738 := (publicBatchWrapper89_copies2898 a h).2.2.1
  have c92739 := (publicBatchWrapper89_copies2898 a h).2.2.2.1
  have c92740 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.1
  have c92741 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_8 := arithEq_of_rows h (row := 1512) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_8
  simp only [← c92736, ← c92737, ← c92738] at e_1512_8
  have e_1512_9 := arithEq_of_rows h (row := 1512) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_9
  simp only [← c92739, ← c92740, k2, ← c92741] at e_1512_9
  have hr := e_1512_9
  simp only [e_1512_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14610)) := by
  have c92742 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.1
  have c92743 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.1
  have c92744 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.1
  have c92745 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.1
  have c92746 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92747 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_10 := arithEq_of_rows h (row := 1512) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_10
  simp only [← c92742, ← c92743, ← c92744] at e_1512_10
  have e_1512_11 := arithEq_of_rows h (row := 1512) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_11
  simp only [← c92745, ← c92746, k2, ← c92747] at e_1512_11
  have hr := e_1512_11
  simp only [e_1512_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14611)) := by
  have c92748 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92749 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92750 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92751 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92752 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92753 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_12 := arithEq_of_rows h (row := 1512) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_12
  simp only [← c92748, ← c92749, ← c92750] at e_1512_12
  have e_1512_13 := arithEq_of_rows h (row := 1512) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_13
  simp only [← c92751, ← c92752, k2, ← c92753] at e_1512_13
  have hr := e_1512_13
  simp only [e_1512_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14612)) := by
  have c92754 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92755 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92756 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92757 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92758 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92759 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_14 := arithEq_of_rows h (row := 1512) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_14
  simp only [← c92754, ← c92755, ← c92756] at e_1512_14
  have e_1512_15 := arithEq_of_rows h (row := 1512) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_15
  simp only [← c92757, ← c92758, k2, ← c92759] at e_1512_15
  have hr := e_1512_15
  simp only [e_1512_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14613)) := by
  have c92760 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92761 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92762 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92763 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92764 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92765 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_16 := arithEq_of_rows h (row := 1512) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_16
  simp only [← c92760, ← c92761, ← c92762] at e_1512_16
  have e_1512_17 := arithEq_of_rows h (row := 1512) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_17
  simp only [← c92763, ← c92764, k2, ← c92765] at e_1512_17
  have hr := e_1512_17
  simp only [e_1512_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14614)) := by
  have c92766 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92767 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92768 := (publicBatchWrapper89_copies2899 a h).1
  have c92769 := (publicBatchWrapper89_copies2899 a h).2.1
  have c92770 := (publicBatchWrapper89_copies2899 a h).2.2.1
  have c92771 := (publicBatchWrapper89_copies2899 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_18 := arithEq_of_rows h (row := 1512) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_18
  simp only [← c92766, ← c92767, ← c92768] at e_1512_18
  have e_1512_19 := arithEq_of_rows h (row := 1512) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_19
  simp only [← c92769, ← c92770, k2, ← c92771] at e_1512_19
  have hr := e_1512_19
  simp only [e_1512_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14615)) := by
  have c92772 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.1
  have c92773 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.1
  have c92774 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.1
  have c92775 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.1
  have c92776 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.1
  have c92777 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_0 := arithEq_of_rows h (row := 1513) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_0
  simp only [← c92772, ← c92773, ← c92774] at e_1513_0
  have e_1513_1 := arithEq_of_rows h (row := 1513) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_1
  simp only [← c92775, ← c92776, k2, ← c92777] at e_1513_1
  have hr := e_1513_1
  simp only [e_1513_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14616)) := by
  have c92778 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92779 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92780 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92781 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92782 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92783 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_2 := arithEq_of_rows h (row := 1513) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_2
  simp only [← c92778, ← c92779, ← c92780] at e_1513_2
  have e_1513_3 := arithEq_of_rows h (row := 1513) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_3
  simp only [← c92781, ← c92782, k2, ← c92783] at e_1513_3
  have hr := e_1513_3
  simp only [e_1513_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14617)) := by
  have c92784 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92785 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92786 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92787 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92788 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92789 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_4 := arithEq_of_rows h (row := 1513) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_4
  simp only [← c92784, ← c92785, ← c92786] at e_1513_4
  have e_1513_5 := arithEq_of_rows h (row := 1513) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_5
  simp only [← c92787, ← c92788, k2, ← c92789] at e_1513_5
  have hr := e_1513_5
  simp only [e_1513_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14618)) := by
  have c92790 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92791 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92792 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92793 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92794 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92795 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_6 := arithEq_of_rows h (row := 1513) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_6
  simp only [← c92790, ← c92791, ← c92792] at e_1513_6
  have e_1513_7 := arithEq_of_rows h (row := 1513) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_7
  simp only [← c92793, ← c92794, k2, ← c92795] at e_1513_7
  have hr := e_1513_7
  simp only [e_1513_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14619)) := by
  have c92796 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92797 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92798 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92799 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92800 := (publicBatchWrapper89_copies2900 a h).1
  have c92801 := (publicBatchWrapper89_copies2900 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_8 := arithEq_of_rows h (row := 1513) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_8
  simp only [← c92796, ← c92797, ← c92798] at e_1513_8
  have e_1513_9 := arithEq_of_rows h (row := 1513) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_9
  simp only [← c92799, ← c92800, k2, ← c92801] at e_1513_9
  have hr := e_1513_9
  simp only [e_1513_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14620)) := by
  have c92802 := (publicBatchWrapper89_copies2900 a h).2.2.1
  have c92803 := (publicBatchWrapper89_copies2900 a h).2.2.2.1
  have c92804 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.1
  have c92805 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.1
  have c92806 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.1
  have c92807 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_10 := arithEq_of_rows h (row := 1513) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_10
  simp only [← c92802, ← c92803, ← c92804] at e_1513_10
  have e_1513_11 := arithEq_of_rows h (row := 1513) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_11
  simp only [← c92805, ← c92806, k2, ← c92807] at e_1513_11
  have hr := e_1513_11
  simp only [e_1513_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14621)) := by
  have c92808 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.1
  have c92809 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.1
  have c92810 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92811 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92812 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92813 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_12 := arithEq_of_rows h (row := 1513) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_12
  simp only [← c92808, ← c92809, ← c92810] at e_1513_12
  have e_1513_13 := arithEq_of_rows h (row := 1513) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_13
  simp only [← c92811, ← c92812, k2, ← c92813] at e_1513_13
  have hr := e_1513_13
  simp only [e_1513_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14622)) := by
  have c92814 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92815 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92816 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92817 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92818 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92819 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_14 := arithEq_of_rows h (row := 1513) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_14
  simp only [← c92814, ← c92815, ← c92816] at e_1513_14
  have e_1513_15 := arithEq_of_rows h (row := 1513) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_15
  simp only [← c92817, ← c92818, k2, ← c92819] at e_1513_15
  have hr := e_1513_15
  simp only [e_1513_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14623)) := by
  have c92820 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92821 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92822 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92823 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92824 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92825 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_16 := arithEq_of_rows h (row := 1513) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_16
  simp only [← c92820, ← c92821, ← c92822] at e_1513_16
  have e_1513_17 := arithEq_of_rows h (row := 1513) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_17
  simp only [← c92823, ← c92824, k2, ← c92825] at e_1513_17
  have hr := e_1513_17
  simp only [e_1513_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14624)) := by
  have c92826 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92827 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92828 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92829 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92830 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92831 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_18 := arithEq_of_rows h (row := 1513) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_18
  simp only [← c92826, ← c92827, ← c92828] at e_1513_18
  have e_1513_19 := arithEq_of_rows h (row := 1513) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_19
  simp only [← c92829, ← c92830, k2, ← c92831] at e_1513_19
  have hr := e_1513_19
  simp only [e_1513_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14625)) := by
  have c92832 := (publicBatchWrapper89_copies2901 a h).1
  have c92833 := (publicBatchWrapper89_copies2901 a h).2.1
  have c92834 := (publicBatchWrapper89_copies2901 a h).2.2.1
  have c92835 := (publicBatchWrapper89_copies2901 a h).2.2.2.1
  have c92836 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.1
  have c92837 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_0 := arithEq_of_rows h (row := 1514) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_0
  simp only [← c92832, ← c92833, ← c92834] at e_1514_0
  have e_1514_1 := arithEq_of_rows h (row := 1514) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_1
  simp only [← c92835, ← c92836, k2, ← c92837] at e_1514_1
  have hr := e_1514_1
  simp only [e_1514_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14626)) := by
  have c92838 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.1
  have c92839 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.1
  have c92840 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.1
  have c92841 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.1
  have c92842 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92843 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_2 := arithEq_of_rows h (row := 1514) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_2
  simp only [← c92838, ← c92839, ← c92840] at e_1514_2
  have e_1514_3 := arithEq_of_rows h (row := 1514) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_3
  simp only [← c92841, ← c92842, k2, ← c92843] at e_1514_3
  have hr := e_1514_3
  simp only [e_1514_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14627)) := by
  have c92844 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92845 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92846 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92847 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92848 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92849 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_4 := arithEq_of_rows h (row := 1514) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_4
  simp only [← c92844, ← c92845, ← c92846] at e_1514_4
  have e_1514_5 := arithEq_of_rows h (row := 1514) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_5
  simp only [← c92847, ← c92848, k2, ← c92849] at e_1514_5
  have hr := e_1514_5
  simp only [e_1514_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14628)) := by
  have c92850 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92851 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92852 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92853 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92854 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92855 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_6 := arithEq_of_rows h (row := 1514) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_6
  simp only [← c92850, ← c92851, ← c92852] at e_1514_6
  have e_1514_7 := arithEq_of_rows h (row := 1514) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_7
  simp only [← c92853, ← c92854, k2, ← c92855] at e_1514_7
  have hr := e_1514_7
  simp only [e_1514_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14629)) := by
  have c92856 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92857 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92858 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92859 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92860 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92861 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_8 := arithEq_of_rows h (row := 1514) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_8
  simp only [← c92856, ← c92857, ← c92858] at e_1514_8
  have e_1514_9 := arithEq_of_rows h (row := 1514) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_9
  simp only [← c92859, ← c92860, k2, ← c92861] at e_1514_9
  have hr := e_1514_9
  simp only [e_1514_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14630)) := by
  have c92862 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92863 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92864 := (publicBatchWrapper89_copies2902 a h).1
  have c92865 := (publicBatchWrapper89_copies2902 a h).2.1
  have c92866 := (publicBatchWrapper89_copies2902 a h).2.2.1
  have c92867 := (publicBatchWrapper89_copies2902 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_10 := arithEq_of_rows h (row := 1514) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_10
  simp only [← c92862, ← c92863, ← c92864] at e_1514_10
  have e_1514_11 := arithEq_of_rows h (row := 1514) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_11
  simp only [← c92865, ← c92866, k2, ← c92867] at e_1514_11
  have hr := e_1514_11
  simp only [e_1514_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14631)) := by
  have c92868 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.1
  have c92869 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.1
  have c92870 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.1
  have c92871 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.1
  have c92872 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.1
  have c92873 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_12 := arithEq_of_rows h (row := 1514) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_12
  simp only [← c92868, ← c92869, ← c92870] at e_1514_12
  have e_1514_13 := arithEq_of_rows h (row := 1514) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_13
  simp only [← c92871, ← c92872, k2, ← c92873] at e_1514_13
  have hr := e_1514_13
  simp only [e_1514_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14632)) := by
  have c92874 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92875 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92876 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92877 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92878 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92879 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_14 := arithEq_of_rows h (row := 1514) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_14
  simp only [← c92874, ← c92875, ← c92876] at e_1514_14
  have e_1514_15 := arithEq_of_rows h (row := 1514) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_15
  simp only [← c92877, ← c92878, k2, ← c92879] at e_1514_15
  have hr := e_1514_15
  simp only [e_1514_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14633)) := by
  have c92880 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92881 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92882 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92883 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92884 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92885 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_16 := arithEq_of_rows h (row := 1514) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_16
  simp only [← c92880, ← c92881, ← c92882] at e_1514_16
  have e_1514_17 := arithEq_of_rows h (row := 1514) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_17
  simp only [← c92883, ← c92884, k2, ← c92885] at e_1514_17
  have hr := e_1514_17
  simp only [e_1514_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14634)) := by
  have c92886 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92887 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92888 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92889 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92890 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92891 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_18 := arithEq_of_rows h (row := 1514) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_18
  simp only [← c92886, ← c92887, ← c92888] at e_1514_18
  have e_1514_19 := arithEq_of_rows h (row := 1514) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_19
  simp only [← c92889, ← c92890, k2, ← c92891] at e_1514_19
  have hr := e_1514_19
  simp only [e_1514_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14635)) := by
  have c92892 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92893 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92894 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92895 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92896 := (publicBatchWrapper89_copies2903 a h).1
  have c92897 := (publicBatchWrapper89_copies2903 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_0 := arithEq_of_rows h (row := 1515) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_0
  simp only [← c92892, ← c92893, ← c92894] at e_1515_0
  have e_1515_1 := arithEq_of_rows h (row := 1515) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_1
  simp only [← c92895, ← c92896, k2, ← c92897] at e_1515_1
  have hr := e_1515_1
  simp only [e_1515_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14636)) := by
  have c92898 := (publicBatchWrapper89_copies2903 a h).2.2.1
  have c92899 := (publicBatchWrapper89_copies2903 a h).2.2.2.1
  have c92900 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.1
  have c92901 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.1
  have c92902 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.1
  have c92903 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_2 := arithEq_of_rows h (row := 1515) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_2
  simp only [← c92898, ← c92899, ← c92900] at e_1515_2
  have e_1515_3 := arithEq_of_rows h (row := 1515) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_3
  simp only [← c92901, ← c92902, k2, ← c92903] at e_1515_3
  have hr := e_1515_3
  simp only [e_1515_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14637)) := by
  have c92904 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.1
  have c92905 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.1
  have c92906 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92907 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92908 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92909 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_4 := arithEq_of_rows h (row := 1515) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_4
  simp only [← c92904, ← c92905, ← c92906] at e_1515_4
  have e_1515_5 := arithEq_of_rows h (row := 1515) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_5
  simp only [← c92907, ← c92908, k2, ← c92909] at e_1515_5
  have hr := e_1515_5
  simp only [e_1515_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14638)) := by
  have c92910 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92911 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92912 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92913 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92914 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92915 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_6 := arithEq_of_rows h (row := 1515) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_6
  simp only [← c92910, ← c92911, ← c92912] at e_1515_6
  have e_1515_7 := arithEq_of_rows h (row := 1515) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_7
  simp only [← c92913, ← c92914, k2, ← c92915] at e_1515_7
  have hr := e_1515_7
  simp only [e_1515_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14639)) := by
  have c92916 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92917 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92918 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92919 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92920 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92921 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_8 := arithEq_of_rows h (row := 1515) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_8
  simp only [← c92916, ← c92917, ← c92918] at e_1515_8
  have e_1515_9 := arithEq_of_rows h (row := 1515) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_9
  simp only [← c92919, ← c92920, k2, ← c92921] at e_1515_9
  have hr := e_1515_9
  simp only [e_1515_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14640)) := by
  have c92922 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92923 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92924 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92925 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92926 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92927 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_10 := arithEq_of_rows h (row := 1515) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_10
  simp only [← c92922, ← c92923, ← c92924] at e_1515_10
  have e_1515_11 := arithEq_of_rows h (row := 1515) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_11
  simp only [← c92925, ← c92926, k2, ← c92927] at e_1515_11
  have hr := e_1515_11
  simp only [e_1515_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14641)) := by
  have c92928 := (publicBatchWrapper89_copies2904 a h).1
  have c92929 := (publicBatchWrapper89_copies2904 a h).2.1
  have c92930 := (publicBatchWrapper89_copies2904 a h).2.2.1
  have c92931 := (publicBatchWrapper89_copies2904 a h).2.2.2.1
  have c92932 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.1
  have c92933 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_12 := arithEq_of_rows h (row := 1515) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_12
  simp only [← c92928, ← c92929, ← c92930] at e_1515_12
  have e_1515_13 := arithEq_of_rows h (row := 1515) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_13
  simp only [← c92931, ← c92932, k2, ← c92933] at e_1515_13
  have hr := e_1515_13
  simp only [e_1515_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14642)) := by
  have c92934 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.1
  have c92935 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.1
  have c92936 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.1
  have c92937 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.1
  have c92938 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92939 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_14 := arithEq_of_rows h (row := 1515) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_14
  simp only [← c92934, ← c92935, ← c92936] at e_1515_14
  have e_1515_15 := arithEq_of_rows h (row := 1515) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_15
  simp only [← c92937, ← c92938, k2, ← c92939] at e_1515_15
  have hr := e_1515_15
  simp only [e_1515_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14643)) := by
  have c92940 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92941 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92942 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92943 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92944 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92945 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_16 := arithEq_of_rows h (row := 1515) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_16
  simp only [← c92940, ← c92941, ← c92942] at e_1515_16
  have e_1515_17 := arithEq_of_rows h (row := 1515) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_17
  simp only [← c92943, ← c92944, k2, ← c92945] at e_1515_17
  have hr := e_1515_17
  simp only [e_1515_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14644)) := by
  have c92946 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92947 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92948 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92949 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92950 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92951 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_18 := arithEq_of_rows h (row := 1515) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_18
  simp only [← c92946, ← c92947, ← c92948] at e_1515_18
  have e_1515_19 := arithEq_of_rows h (row := 1515) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_19
  simp only [← c92949, ← c92950, k2, ← c92951] at e_1515_19
  have hr := e_1515_19
  simp only [e_1515_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14645)) := by
  have c92952 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92953 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92954 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92955 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92956 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92957 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_0 := arithEq_of_rows h (row := 1516) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_0
  simp only [← c92952, ← c92953, ← c92954] at e_1516_0
  have e_1516_1 := arithEq_of_rows h (row := 1516) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_1
  simp only [← c92955, ← c92956, k2, ← c92957] at e_1516_1
  have hr := e_1516_1
  simp only [e_1516_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14646)) := by
  have c92958 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92959 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92960 := (publicBatchWrapper89_copies2905 a h).1
  have c92961 := (publicBatchWrapper89_copies2905 a h).2.1
  have c92962 := (publicBatchWrapper89_copies2905 a h).2.2.1
  have c92963 := (publicBatchWrapper89_copies2905 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_2 := arithEq_of_rows h (row := 1516) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_2
  simp only [← c92958, ← c92959, ← c92960] at e_1516_2
  have e_1516_3 := arithEq_of_rows h (row := 1516) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_3
  simp only [← c92961, ← c92962, k2, ← c92963] at e_1516_3
  have hr := e_1516_3
  simp only [e_1516_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14647)) := by
  have c92964 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.1
  have c92965 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.1
  have c92966 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.1
  have c92967 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.1
  have c92968 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.1
  have c92969 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_4 := arithEq_of_rows h (row := 1516) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_4
  simp only [← c92964, ← c92965, ← c92966] at e_1516_4
  have e_1516_5 := arithEq_of_rows h (row := 1516) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_5
  simp only [← c92967, ← c92968, k2, ← c92969] at e_1516_5
  have hr := e_1516_5
  simp only [e_1516_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14648)) := by
  have c92970 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92971 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92972 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92973 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92974 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92975 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_6 := arithEq_of_rows h (row := 1516) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_6
  simp only [← c92970, ← c92971, ← c92972] at e_1516_6
  have e_1516_7 := arithEq_of_rows h (row := 1516) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_7
  simp only [← c92973, ← c92974, k2, ← c92975] at e_1516_7
  have hr := e_1516_7
  simp only [e_1516_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14649)) := by
  have c92976 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92977 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92978 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92979 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92980 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92981 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_8 := arithEq_of_rows h (row := 1516) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_8
  simp only [← c92976, ← c92977, ← c92978] at e_1516_8
  have e_1516_9 := arithEq_of_rows h (row := 1516) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_9
  simp only [← c92979, ← c92980, k2, ← c92981] at e_1516_9
  have hr := e_1516_9
  simp only [e_1516_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14650)) := by
  have c92982 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92983 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92984 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92985 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92986 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92987 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_10 := arithEq_of_rows h (row := 1516) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_10
  simp only [← c92982, ← c92983, ← c92984] at e_1516_10
  have e_1516_11 := arithEq_of_rows h (row := 1516) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_11
  simp only [← c92985, ← c92986, k2, ← c92987] at e_1516_11
  have hr := e_1516_11
  simp only [e_1516_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14651)) := by
  have c92988 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92989 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92990 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92991 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92992 := (publicBatchWrapper89_copies2906 a h).1
  have c92993 := (publicBatchWrapper89_copies2906 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_12 := arithEq_of_rows h (row := 1516) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_12
  simp only [← c92988, ← c92989, ← c92990] at e_1516_12
  have e_1516_13 := arithEq_of_rows h (row := 1516) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_13
  simp only [← c92991, ← c92992, k2, ← c92993] at e_1516_13
  have hr := e_1516_13
  simp only [e_1516_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14652)) := by
  have c92994 := (publicBatchWrapper89_copies2906 a h).2.2.1
  have c92995 := (publicBatchWrapper89_copies2906 a h).2.2.2.1
  have c92996 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.1
  have c92997 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.1
  have c92998 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.1
  have c92999 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_14 := arithEq_of_rows h (row := 1516) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_14
  simp only [← c92994, ← c92995, ← c92996] at e_1516_14
  have e_1516_15 := arithEq_of_rows h (row := 1516) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_15
  simp only [← c92997, ← c92998, k2, ← c92999] at e_1516_15
  have hr := e_1516_15
  simp only [e_1516_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14653)) := by
  have c93000 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.1
  have c93001 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.1
  have c93002 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93003 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93004 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93005 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_16 := arithEq_of_rows h (row := 1516) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_16
  simp only [← c93000, ← c93001, ← c93002] at e_1516_16
  have e_1516_17 := arithEq_of_rows h (row := 1516) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_17
  simp only [← c93003, ← c93004, k2, ← c93005] at e_1516_17
  have hr := e_1516_17
  simp only [e_1516_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14654)) := by
  have c93006 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93007 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93008 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93009 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93010 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93011 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_18 := arithEq_of_rows h (row := 1516) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_18
  simp only [← c93006, ← c93007, ← c93008] at e_1516_18
  have e_1516_19 := arithEq_of_rows h (row := 1516) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_19
  simp only [← c93009, ← c93010, k2, ← c93011] at e_1516_19
  have hr := e_1516_19
  simp only [e_1516_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14655)) := by
  have c93012 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93013 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93014 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93015 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93016 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93017 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_0 := arithEq_of_rows h (row := 1517) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_0
  simp only [← c93012, ← c93013, ← c93014] at e_1517_0
  have e_1517_1 := arithEq_of_rows h (row := 1517) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_1
  simp only [← c93015, ← c93016, k2, ← c93017] at e_1517_1
  have hr := e_1517_1
  simp only [e_1517_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14656)) := by
  have c93018 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93019 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93020 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93021 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93022 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93023 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_2 := arithEq_of_rows h (row := 1517) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_2
  simp only [← c93018, ← c93019, ← c93020] at e_1517_2
  have e_1517_3 := arithEq_of_rows h (row := 1517) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_3
  simp only [← c93021, ← c93022, k2, ← c93023] at e_1517_3
  have hr := e_1517_3
  simp only [e_1517_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14657)) := by
  have c93024 := (publicBatchWrapper89_copies2907 a h).1
  have c93025 := (publicBatchWrapper89_copies2907 a h).2.1
  have c93026 := (publicBatchWrapper89_copies2907 a h).2.2.1
  have c93027 := (publicBatchWrapper89_copies2907 a h).2.2.2.1
  have c93028 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.1
  have c93029 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_4 := arithEq_of_rows h (row := 1517) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_4
  simp only [← c93024, ← c93025, ← c93026] at e_1517_4
  have e_1517_5 := arithEq_of_rows h (row := 1517) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_5
  simp only [← c93027, ← c93028, k2, ← c93029] at e_1517_5
  have hr := e_1517_5
  simp only [e_1517_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14658)) := by
  have c93030 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.1
  have c93031 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.1
  have c93032 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.1
  have c93033 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.1
  have c93034 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93035 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_6 := arithEq_of_rows h (row := 1517) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_6
  simp only [← c93030, ← c93031, ← c93032] at e_1517_6
  have e_1517_7 := arithEq_of_rows h (row := 1517) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_7
  simp only [← c93033, ← c93034, k2, ← c93035] at e_1517_7
  have hr := e_1517_7
  simp only [e_1517_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14659)) := by
  have c93036 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93037 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93038 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93039 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93040 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93041 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_8 := arithEq_of_rows h (row := 1517) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_8
  simp only [← c93036, ← c93037, ← c93038] at e_1517_8
  have e_1517_9 := arithEq_of_rows h (row := 1517) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_9
  simp only [← c93039, ← c93040, k2, ← c93041] at e_1517_9
  have hr := e_1517_9
  simp only [e_1517_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14660)) := by
  have c93042 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93043 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93044 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93045 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93046 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93047 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_10 := arithEq_of_rows h (row := 1517) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_10
  simp only [← c93042, ← c93043, ← c93044] at e_1517_10
  have e_1517_11 := arithEq_of_rows h (row := 1517) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_11
  simp only [← c93045, ← c93046, k2, ← c93047] at e_1517_11
  have hr := e_1517_11
  simp only [e_1517_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14661)) := by
  have c93048 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93049 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93050 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93051 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93052 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93053 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_12 := arithEq_of_rows h (row := 1517) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_12
  simp only [← c93048, ← c93049, ← c93050] at e_1517_12
  have e_1517_13 := arithEq_of_rows h (row := 1517) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_13
  simp only [← c93051, ← c93052, k2, ← c93053] at e_1517_13
  have hr := e_1517_13
  simp only [e_1517_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14662)) := by
  have c93054 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93055 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93056 := (publicBatchWrapper89_copies2908 a h).1
  have c93057 := (publicBatchWrapper89_copies2908 a h).2.1
  have c93058 := (publicBatchWrapper89_copies2908 a h).2.2.1
  have c93059 := (publicBatchWrapper89_copies2908 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_14 := arithEq_of_rows h (row := 1517) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_14
  simp only [← c93054, ← c93055, ← c93056] at e_1517_14
  have e_1517_15 := arithEq_of_rows h (row := 1517) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_15
  simp only [← c93057, ← c93058, k2, ← c93059] at e_1517_15
  have hr := e_1517_15
  simp only [e_1517_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14663)) := by
  have c93060 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.1
  have c93061 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.1
  have c93062 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.1
  have c93063 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.1
  have c93064 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.1
  have c93065 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_16 := arithEq_of_rows h (row := 1517) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_16
  simp only [← c93060, ← c93061, ← c93062] at e_1517_16
  have e_1517_17 := arithEq_of_rows h (row := 1517) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_17
  simp only [← c93063, ← c93064, k2, ← c93065] at e_1517_17
  have hr := e_1517_17
  simp only [e_1517_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14664)) := by
  have c93066 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93067 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93068 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93069 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93070 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93071 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_18 := arithEq_of_rows h (row := 1517) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_18
  simp only [← c93066, ← c93067, ← c93068] at e_1517_18
  have e_1517_19 := arithEq_of_rows h (row := 1517) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_19
  simp only [← c93069, ← c93070, k2, ← c93071] at e_1517_19
  have hr := e_1517_19
  simp only [e_1517_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14665)) := by
  have c93072 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93073 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93074 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93075 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93076 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93077 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_0 := arithEq_of_rows h (row := 1518) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_0
  simp only [← c93072, ← c93073, ← c93074] at e_1518_0
  have e_1518_1 := arithEq_of_rows h (row := 1518) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_1
  simp only [← c93075, ← c93076, k2, ← c93077] at e_1518_1
  have hr := e_1518_1
  simp only [e_1518_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14666)) := by
  have c93078 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93079 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93080 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93081 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93082 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93083 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_2 := arithEq_of_rows h (row := 1518) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_2
  simp only [← c93078, ← c93079, ← c93080] at e_1518_2
  have e_1518_3 := arithEq_of_rows h (row := 1518) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_3
  simp only [← c93081, ← c93082, k2, ← c93083] at e_1518_3
  have hr := e_1518_3
  simp only [e_1518_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14667)) := by
  have c93084 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93085 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93086 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93087 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93088 := (publicBatchWrapper89_copies2909 a h).1
  have c93089 := (publicBatchWrapper89_copies2909 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_4 := arithEq_of_rows h (row := 1518) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_4
  simp only [← c93084, ← c93085, ← c93086] at e_1518_4
  have e_1518_5 := arithEq_of_rows h (row := 1518) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_5
  simp only [← c93087, ← c93088, k2, ← c93089] at e_1518_5
  have hr := e_1518_5
  simp only [e_1518_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14668)) := by
  have c93090 := (publicBatchWrapper89_copies2909 a h).2.2.1
  have c93091 := (publicBatchWrapper89_copies2909 a h).2.2.2.1
  have c93092 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.1
  have c93093 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.1
  have c93094 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.1
  have c93095 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_6 := arithEq_of_rows h (row := 1518) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_6
  simp only [← c93090, ← c93091, ← c93092] at e_1518_6
  have e_1518_7 := arithEq_of_rows h (row := 1518) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_7
  simp only [← c93093, ← c93094, k2, ← c93095] at e_1518_7
  have hr := e_1518_7
  simp only [e_1518_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14669)) := by
  have c93096 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.1
  have c93097 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.1
  have c93098 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93099 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93100 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93101 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_8 := arithEq_of_rows h (row := 1518) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_8
  simp only [← c93096, ← c93097, ← c93098] at e_1518_8
  have e_1518_9 := arithEq_of_rows h (row := 1518) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_9
  simp only [← c93099, ← c93100, k2, ← c93101] at e_1518_9
  have hr := e_1518_9
  simp only [e_1518_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14670)) := by
  have c93102 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93103 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93104 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93105 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93106 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93107 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_10 := arithEq_of_rows h (row := 1518) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_10
  simp only [← c93102, ← c93103, ← c93104] at e_1518_10
  have e_1518_11 := arithEq_of_rows h (row := 1518) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_11
  simp only [← c93105, ← c93106, k2, ← c93107] at e_1518_11
  have hr := e_1518_11
  simp only [e_1518_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14671)) := by
  have c93108 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93109 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93110 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93111 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93112 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93113 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_12 := arithEq_of_rows h (row := 1518) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_12
  simp only [← c93108, ← c93109, ← c93110] at e_1518_12
  have e_1518_13 := arithEq_of_rows h (row := 1518) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_13
  simp only [← c93111, ← c93112, k2, ← c93113] at e_1518_13
  have hr := e_1518_13
  simp only [e_1518_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14672)) := by
  have c93114 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93115 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93116 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93117 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93118 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93119 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_14 := arithEq_of_rows h (row := 1518) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_14
  simp only [← c93114, ← c93115, ← c93116] at e_1518_14
  have e_1518_15 := arithEq_of_rows h (row := 1518) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_15
  simp only [← c93117, ← c93118, k2, ← c93119] at e_1518_15
  have hr := e_1518_15
  simp only [e_1518_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14673)) := by
  have c93120 := (publicBatchWrapper89_copies2910 a h).1
  have c93121 := (publicBatchWrapper89_copies2910 a h).2.1
  have c93122 := (publicBatchWrapper89_copies2910 a h).2.2.1
  have c93123 := (publicBatchWrapper89_copies2910 a h).2.2.2.1
  have c93124 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.1
  have c93125 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_16 := arithEq_of_rows h (row := 1518) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_16
  simp only [← c93120, ← c93121, ← c93122] at e_1518_16
  have e_1518_17 := arithEq_of_rows h (row := 1518) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_17
  simp only [← c93123, ← c93124, k2, ← c93125] at e_1518_17
  have hr := e_1518_17
  simp only [e_1518_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14674)) := by
  have c93126 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.1
  have c93127 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.1
  have c93128 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.1
  have c93129 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.1
  have c93130 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93131 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_18 := arithEq_of_rows h (row := 1518) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_18
  simp only [← c93126, ← c93127, ← c93128] at e_1518_18
  have e_1518_19 := arithEq_of_rows h (row := 1518) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_19
  simp only [← c93129, ← c93130, k2, ← c93131] at e_1518_19
  have hr := e_1518_19
  simp only [e_1518_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14675)) := by
  have c93132 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93133 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93134 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93135 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93136 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93137 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_0 := arithEq_of_rows h (row := 1519) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_0
  simp only [← c93132, ← c93133, ← c93134] at e_1519_0
  have e_1519_1 := arithEq_of_rows h (row := 1519) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_1
  simp only [← c93135, ← c93136, k2, ← c93137] at e_1519_1
  have hr := e_1519_1
  simp only [e_1519_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14676)) := by
  have c93138 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93139 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93140 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93141 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93142 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93143 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_2 := arithEq_of_rows h (row := 1519) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_2
  simp only [← c93138, ← c93139, ← c93140] at e_1519_2
  have e_1519_3 := arithEq_of_rows h (row := 1519) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_3
  simp only [← c93141, ← c93142, k2, ← c93143] at e_1519_3
  have hr := e_1519_3
  simp only [e_1519_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14677)) := by
  have c93144 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93145 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93146 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93147 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93148 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93149 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_4 := arithEq_of_rows h (row := 1519) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_4
  simp only [← c93144, ← c93145, ← c93146] at e_1519_4
  have e_1519_5 := arithEq_of_rows h (row := 1519) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_5
  simp only [← c93147, ← c93148, k2, ← c93149] at e_1519_5
  have hr := e_1519_5
  simp only [e_1519_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14678)) := by
  have c93150 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93151 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93152 := (publicBatchWrapper89_copies2911 a h).1
  have c93153 := (publicBatchWrapper89_copies2911 a h).2.1
  have c93154 := (publicBatchWrapper89_copies2911 a h).2.2.1
  have c93155 := (publicBatchWrapper89_copies2911 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_6 := arithEq_of_rows h (row := 1519) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_6
  simp only [← c93150, ← c93151, ← c93152] at e_1519_6
  have e_1519_7 := arithEq_of_rows h (row := 1519) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_7
  simp only [← c93153, ← c93154, k2, ← c93155] at e_1519_7
  have hr := e_1519_7
  simp only [e_1519_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14679)) := by
  have c93156 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.1
  have c93157 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.1
  have c93158 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.1
  have c93159 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.1
  have c93160 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.1
  have c93161 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_8 := arithEq_of_rows h (row := 1519) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_8
  simp only [← c93156, ← c93157, ← c93158] at e_1519_8
  have e_1519_9 := arithEq_of_rows h (row := 1519) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_9
  simp only [← c93159, ← c93160, k2, ← c93161] at e_1519_9
  have hr := e_1519_9
  simp only [e_1519_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14680)) := by
  have c93162 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93163 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93164 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93165 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93166 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93167 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_10 := arithEq_of_rows h (row := 1519) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_10
  simp only [← c93162, ← c93163, ← c93164] at e_1519_10
  have e_1519_11 := arithEq_of_rows h (row := 1519) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_11
  simp only [← c93165, ← c93166, k2, ← c93167] at e_1519_11
  have hr := e_1519_11
  simp only [e_1519_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14681)) := by
  have c93168 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93169 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93170 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93171 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93172 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93173 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_12 := arithEq_of_rows h (row := 1519) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_12
  simp only [← c93168, ← c93169, ← c93170] at e_1519_12
  have e_1519_13 := arithEq_of_rows h (row := 1519) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_13
  simp only [← c93171, ← c93172, k2, ← c93173] at e_1519_13
  have hr := e_1519_13
  simp only [e_1519_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14682)) := by
  have c93174 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93175 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93176 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93177 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93178 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93179 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_14 := arithEq_of_rows h (row := 1519) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_14
  simp only [← c93174, ← c93175, ← c93176] at e_1519_14
  have e_1519_15 := arithEq_of_rows h (row := 1519) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_15
  simp only [← c93177, ← c93178, k2, ← c93179] at e_1519_15
  have hr := e_1519_15
  simp only [e_1519_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14683)) := by
  have c93180 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93181 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93182 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93183 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93184 := (publicBatchWrapper89_copies2912 a h).1
  have c93185 := (publicBatchWrapper89_copies2912 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_16 := arithEq_of_rows h (row := 1519) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_16
  simp only [← c93180, ← c93181, ← c93182] at e_1519_16
  have e_1519_17 := arithEq_of_rows h (row := 1519) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_17
  simp only [← c93183, ← c93184, k2, ← c93185] at e_1519_17
  have hr := e_1519_17
  simp only [e_1519_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14684)) := by
  have c93186 := (publicBatchWrapper89_copies2912 a h).2.2.1
  have c93187 := (publicBatchWrapper89_copies2912 a h).2.2.2.1
  have c93188 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.1
  have c93189 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.1
  have c93190 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.1
  have c93191 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_18 := arithEq_of_rows h (row := 1519) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_18
  simp only [← c93186, ← c93187, ← c93188] at e_1519_18
  have e_1519_19 := arithEq_of_rows h (row := 1519) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_19
  simp only [← c93189, ← c93190, k2, ← c93191] at e_1519_19
  have hr := e_1519_19
  simp only [e_1519_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14685)) := by
  have c93192 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.1
  have c93193 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.1
  have c93194 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93195 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93196 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93197 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_0 := arithEq_of_rows h (row := 1520) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_0
  simp only [← c93192, ← c93193, ← c93194] at e_1520_0
  have e_1520_1 := arithEq_of_rows h (row := 1520) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_1
  simp only [← c93195, ← c93196, k2, ← c93197] at e_1520_1
  have hr := e_1520_1
  simp only [e_1520_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14686)) := by
  have c93198 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93199 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93200 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93201 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93202 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93203 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_2 := arithEq_of_rows h (row := 1520) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_2
  simp only [← c93198, ← c93199, ← c93200] at e_1520_2
  have e_1520_3 := arithEq_of_rows h (row := 1520) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_3
  simp only [← c93201, ← c93202, k2, ← c93203] at e_1520_3
  have hr := e_1520_3
  simp only [e_1520_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14687)) := by
  have c93204 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93205 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93206 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93207 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93208 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93209 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_4 := arithEq_of_rows h (row := 1520) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_4
  simp only [← c93204, ← c93205, ← c93206] at e_1520_4
  have e_1520_5 := arithEq_of_rows h (row := 1520) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_5
  simp only [← c93207, ← c93208, k2, ← c93209] at e_1520_5
  have hr := e_1520_5
  simp only [e_1520_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14688)) := by
  have c93210 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93211 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93212 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93213 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93214 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93215 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_6 := arithEq_of_rows h (row := 1520) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_6
  simp only [← c93210, ← c93211, ← c93212] at e_1520_6
  have e_1520_7 := arithEq_of_rows h (row := 1520) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_7
  simp only [← c93213, ← c93214, k2, ← c93215] at e_1520_7
  have hr := e_1520_7
  simp only [e_1520_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14689)) := by
  have c93216 := (publicBatchWrapper89_copies2913 a h).1
  have c93217 := (publicBatchWrapper89_copies2913 a h).2.1
  have c93218 := (publicBatchWrapper89_copies2913 a h).2.2.1
  have c93219 := (publicBatchWrapper89_copies2913 a h).2.2.2.1
  have c93220 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.1
  have c93221 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_8 := arithEq_of_rows h (row := 1520) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_8
  simp only [← c93216, ← c93217, ← c93218] at e_1520_8
  have e_1520_9 := arithEq_of_rows h (row := 1520) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_9
  simp only [← c93219, ← c93220, k2, ← c93221] at e_1520_9
  have hr := e_1520_9
  simp only [e_1520_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14690)) := by
  have c93222 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.1
  have c93223 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.1
  have c93224 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.1
  have c93225 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.1
  have c93226 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93227 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_10 := arithEq_of_rows h (row := 1520) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_10
  simp only [← c93222, ← c93223, ← c93224] at e_1520_10
  have e_1520_11 := arithEq_of_rows h (row := 1520) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_11
  simp only [← c93225, ← c93226, k2, ← c93227] at e_1520_11
  have hr := e_1520_11
  simp only [e_1520_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14691)) := by
  have c93228 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93229 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93230 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93231 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93232 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93233 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_12 := arithEq_of_rows h (row := 1520) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_12
  simp only [← c93228, ← c93229, ← c93230] at e_1520_12
  have e_1520_13 := arithEq_of_rows h (row := 1520) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_13
  simp only [← c93231, ← c93232, k2, ← c93233] at e_1520_13
  have hr := e_1520_13
  simp only [e_1520_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14692)) := by
  have c93234 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93235 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93236 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93237 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93238 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93239 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_14 := arithEq_of_rows h (row := 1520) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_14
  simp only [← c93234, ← c93235, ← c93236] at e_1520_14
  have e_1520_15 := arithEq_of_rows h (row := 1520) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_15
  simp only [← c93237, ← c93238, k2, ← c93239] at e_1520_15
  have hr := e_1520_15
  simp only [e_1520_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14693)) := by
  have c93240 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93241 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93242 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93243 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93244 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93245 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_16 := arithEq_of_rows h (row := 1520) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_16
  simp only [← c93240, ← c93241, ← c93242] at e_1520_16
  have e_1520_17 := arithEq_of_rows h (row := 1520) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_17
  simp only [← c93243, ← c93244, k2, ← c93245] at e_1520_17
  have hr := e_1520_17
  simp only [e_1520_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14694)) := by
  have c93246 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93247 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93248 := (publicBatchWrapper89_copies2914 a h).1
  have c93249 := (publicBatchWrapper89_copies2914 a h).2.1
  have c93250 := (publicBatchWrapper89_copies2914 a h).2.2.1
  have c93251 := (publicBatchWrapper89_copies2914 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_18 := arithEq_of_rows h (row := 1520) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_18
  simp only [← c93246, ← c93247, ← c93248] at e_1520_18
  have e_1520_19 := arithEq_of_rows h (row := 1520) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_19
  simp only [← c93249, ← c93250, k2, ← c93251] at e_1520_19
  have hr := e_1520_19
  simp only [e_1520_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14695)) := by
  have c93252 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.1
  have c93253 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.1
  have c93254 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.1
  have c93255 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.1
  have c93256 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.1
  have c93257 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_0 := arithEq_of_rows h (row := 1521) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_0
  simp only [← c93252, ← c93253, ← c93254] at e_1521_0
  have e_1521_1 := arithEq_of_rows h (row := 1521) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_1
  simp only [← c93255, ← c93256, k2, ← c93257] at e_1521_1
  have hr := e_1521_1
  simp only [e_1521_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14696)) := by
  have c93258 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93259 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93260 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93261 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93262 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93263 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_2 := arithEq_of_rows h (row := 1521) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_2
  simp only [← c93258, ← c93259, ← c93260] at e_1521_2
  have e_1521_3 := arithEq_of_rows h (row := 1521) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_3
  simp only [← c93261, ← c93262, k2, ← c93263] at e_1521_3
  have hr := e_1521_3
  simp only [e_1521_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14697)) := by
  have c93264 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93265 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93266 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93267 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93268 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93269 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_4 := arithEq_of_rows h (row := 1521) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_4
  simp only [← c93264, ← c93265, ← c93266] at e_1521_4
  have e_1521_5 := arithEq_of_rows h (row := 1521) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_5
  simp only [← c93267, ← c93268, k2, ← c93269] at e_1521_5
  have hr := e_1521_5
  simp only [e_1521_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14698)) := by
  have c93270 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93271 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93272 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93273 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93274 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93275 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_6 := arithEq_of_rows h (row := 1521) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_6
  simp only [← c93270, ← c93271, ← c93272] at e_1521_6
  have e_1521_7 := arithEq_of_rows h (row := 1521) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_7
  simp only [← c93273, ← c93274, k2, ← c93275] at e_1521_7
  have hr := e_1521_7
  simp only [e_1521_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14699)) := by
  have c93276 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93277 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93278 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93279 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93280 := (publicBatchWrapper89_copies2915 a h).1
  have c93281 := (publicBatchWrapper89_copies2915 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_8 := arithEq_of_rows h (row := 1521) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_8
  simp only [← c93276, ← c93277, ← c93278] at e_1521_8
  have e_1521_9 := arithEq_of_rows h (row := 1521) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_9
  simp only [← c93279, ← c93280, k2, ← c93281] at e_1521_9
  have hr := e_1521_9
  simp only [e_1521_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14700)) := by
  have c93282 := (publicBatchWrapper89_copies2915 a h).2.2.1
  have c93283 := (publicBatchWrapper89_copies2915 a h).2.2.2.1
  have c93284 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.1
  have c93285 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.1
  have c93286 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.1
  have c93287 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_10 := arithEq_of_rows h (row := 1521) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_10
  simp only [← c93282, ← c93283, ← c93284] at e_1521_10
  have e_1521_11 := arithEq_of_rows h (row := 1521) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_11
  simp only [← c93285, ← c93286, k2, ← c93287] at e_1521_11
  have hr := e_1521_11
  simp only [e_1521_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14701)) := by
  have c93288 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.1
  have c93289 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.1
  have c93290 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93291 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93292 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93293 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_12 := arithEq_of_rows h (row := 1521) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_12
  simp only [← c93288, ← c93289, ← c93290] at e_1521_12
  have e_1521_13 := arithEq_of_rows h (row := 1521) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_13
  simp only [← c93291, ← c93292, k2, ← c93293] at e_1521_13
  have hr := e_1521_13
  simp only [e_1521_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14702)) := by
  have c93294 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93295 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93296 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93297 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93298 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93299 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_14 := arithEq_of_rows h (row := 1521) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_14
  simp only [← c93294, ← c93295, ← c93296] at e_1521_14
  have e_1521_15 := arithEq_of_rows h (row := 1521) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_15
  simp only [← c93297, ← c93298, k2, ← c93299] at e_1521_15
  have hr := e_1521_15
  simp only [e_1521_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14703)) := by
  have c93300 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93301 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93302 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93303 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93304 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93305 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_16 := arithEq_of_rows h (row := 1521) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_16
  simp only [← c93300, ← c93301, ← c93302] at e_1521_16
  have e_1521_17 := arithEq_of_rows h (row := 1521) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_17
  simp only [← c93303, ← c93304, k2, ← c93305] at e_1521_17
  have hr := e_1521_17
  simp only [e_1521_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14704)) := by
  have c93306 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93307 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93308 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93309 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93310 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93311 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_18 := arithEq_of_rows h (row := 1521) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_18
  simp only [← c93306, ← c93307, ← c93308] at e_1521_18
  have e_1521_19 := arithEq_of_rows h (row := 1521) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_19
  simp only [← c93309, ← c93310, k2, ← c93311] at e_1521_19
  have hr := e_1521_19
  simp only [e_1521_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14705)) := by
  have c93312 := (publicBatchWrapper89_copies2916 a h).1
  have c93313 := (publicBatchWrapper89_copies2916 a h).2.1
  have c93314 := (publicBatchWrapper89_copies2916 a h).2.2.1
  have c93315 := (publicBatchWrapper89_copies2916 a h).2.2.2.1
  have c93316 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.1
  have c93317 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_0 := arithEq_of_rows h (row := 1522) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_0
  simp only [← c93312, ← c93313, ← c93314] at e_1522_0
  have e_1522_1 := arithEq_of_rows h (row := 1522) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_1
  simp only [← c93315, ← c93316, k2, ← c93317] at e_1522_1
  have hr := e_1522_1
  simp only [e_1522_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14706)) := by
  have c93318 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.1
  have c93319 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.1
  have c93320 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.1
  have c93321 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.1
  have c93322 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93323 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_2 := arithEq_of_rows h (row := 1522) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_2
  simp only [← c93318, ← c93319, ← c93320] at e_1522_2
  have e_1522_3 := arithEq_of_rows h (row := 1522) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_3
  simp only [← c93321, ← c93322, k2, ← c93323] at e_1522_3
  have hr := e_1522_3
  simp only [e_1522_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14707)) := by
  have c93324 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93325 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93326 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93327 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93328 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93329 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_4 := arithEq_of_rows h (row := 1522) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_4
  simp only [← c93324, ← c93325, ← c93326] at e_1522_4
  have e_1522_5 := arithEq_of_rows h (row := 1522) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_5
  simp only [← c93327, ← c93328, k2, ← c93329] at e_1522_5
  have hr := e_1522_5
  simp only [e_1522_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14708)) := by
  have c93330 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93331 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93332 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93333 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93334 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93335 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_6 := arithEq_of_rows h (row := 1522) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_6
  simp only [← c93330, ← c93331, ← c93332] at e_1522_6
  have e_1522_7 := arithEq_of_rows h (row := 1522) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_7
  simp only [← c93333, ← c93334, k2, ← c93335] at e_1522_7
  have hr := e_1522_7
  simp only [e_1522_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14709)) := by
  have c93336 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93337 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93338 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93339 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93340 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93341 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_8 := arithEq_of_rows h (row := 1522) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_8
  simp only [← c93336, ← c93337, ← c93338] at e_1522_8
  have e_1522_9 := arithEq_of_rows h (row := 1522) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_9
  simp only [← c93339, ← c93340, k2, ← c93341] at e_1522_9
  have hr := e_1522_9
  simp only [e_1522_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14710)) := by
  have c93342 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93343 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93344 := (publicBatchWrapper89_copies2917 a h).1
  have c93345 := (publicBatchWrapper89_copies2917 a h).2.1
  have c93346 := (publicBatchWrapper89_copies2917 a h).2.2.1
  have c93347 := (publicBatchWrapper89_copies2917 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_10 := arithEq_of_rows h (row := 1522) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_10
  simp only [← c93342, ← c93343, ← c93344] at e_1522_10
  have e_1522_11 := arithEq_of_rows h (row := 1522) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_11
  simp only [← c93345, ← c93346, k2, ← c93347] at e_1522_11
  have hr := e_1522_11
  simp only [e_1522_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14711)) := by
  have c93348 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.1
  have c93349 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.1
  have c93350 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.1
  have c93351 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.1
  have c93352 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.1
  have c93353 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_12 := arithEq_of_rows h (row := 1522) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_12
  simp only [← c93348, ← c93349, ← c93350] at e_1522_12
  have e_1522_13 := arithEq_of_rows h (row := 1522) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_13
  simp only [← c93351, ← c93352, k2, ← c93353] at e_1522_13
  have hr := e_1522_13
  simp only [e_1522_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14712)) := by
  have c93354 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93355 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93356 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93357 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93358 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93359 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_14 := arithEq_of_rows h (row := 1522) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_14
  simp only [← c93354, ← c93355, ← c93356] at e_1522_14
  have e_1522_15 := arithEq_of_rows h (row := 1522) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_15
  simp only [← c93357, ← c93358, k2, ← c93359] at e_1522_15
  have hr := e_1522_15
  simp only [e_1522_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14713)) := by
  have c93360 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93361 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93362 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93363 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93364 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93365 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_16 := arithEq_of_rows h (row := 1522) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_16
  simp only [← c93360, ← c93361, ← c93362] at e_1522_16
  have e_1522_17 := arithEq_of_rows h (row := 1522) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_17
  simp only [← c93363, ← c93364, k2, ← c93365] at e_1522_17
  have hr := e_1522_17
  simp only [e_1522_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14714)) := by
  have c93366 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93367 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93368 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93369 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93370 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93371 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_18 := arithEq_of_rows h (row := 1522) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_18
  simp only [← c93366, ← c93367, ← c93368] at e_1522_18
  have e_1522_19 := arithEq_of_rows h (row := 1522) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_19
  simp only [← c93369, ← c93370, k2, ← c93371] at e_1522_19
  have hr := e_1522_19
  simp only [e_1522_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14715)) := by
  have c93372 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93373 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93374 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93375 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93376 := (publicBatchWrapper89_copies2918 a h).1
  have c93377 := (publicBatchWrapper89_copies2918 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_0 := arithEq_of_rows h (row := 1523) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_0
  simp only [← c93372, ← c93373, ← c93374] at e_1523_0
  have e_1523_1 := arithEq_of_rows h (row := 1523) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_1
  simp only [← c93375, ← c93376, k2, ← c93377] at e_1523_1
  have hr := e_1523_1
  simp only [e_1523_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14716)) := by
  have c93378 := (publicBatchWrapper89_copies2918 a h).2.2.1
  have c93379 := (publicBatchWrapper89_copies2918 a h).2.2.2.1
  have c93380 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.1
  have c93381 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.1
  have c93382 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.1
  have c93383 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_2 := arithEq_of_rows h (row := 1523) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_2
  simp only [← c93378, ← c93379, ← c93380] at e_1523_2
  have e_1523_3 := arithEq_of_rows h (row := 1523) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_3
  simp only [← c93381, ← c93382, k2, ← c93383] at e_1523_3
  have hr := e_1523_3
  simp only [e_1523_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14717)) := by
  have c93384 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.1
  have c93385 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.1
  have c93386 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93387 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93388 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93389 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_4 := arithEq_of_rows h (row := 1523) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_4
  simp only [← c93384, ← c93385, ← c93386] at e_1523_4
  have e_1523_5 := arithEq_of_rows h (row := 1523) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_5
  simp only [← c93387, ← c93388, k2, ← c93389] at e_1523_5
  have hr := e_1523_5
  simp only [e_1523_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14718)) := by
  have c93390 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93391 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93392 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93393 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93394 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93395 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_6 := arithEq_of_rows h (row := 1523) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_6
  simp only [← c93390, ← c93391, ← c93392] at e_1523_6
  have e_1523_7 := arithEq_of_rows h (row := 1523) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_7
  simp only [← c93393, ← c93394, k2, ← c93395] at e_1523_7
  have hr := e_1523_7
  simp only [e_1523_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14719)) := by
  have c93396 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93397 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93398 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93399 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93400 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93401 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_8 := arithEq_of_rows h (row := 1523) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_8
  simp only [← c93396, ← c93397, ← c93398] at e_1523_8
  have e_1523_9 := arithEq_of_rows h (row := 1523) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_9
  simp only [← c93399, ← c93400, k2, ← c93401] at e_1523_9
  have hr := e_1523_9
  simp only [e_1523_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14720)) := by
  have c93402 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93403 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93404 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93405 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93406 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93407 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_10 := arithEq_of_rows h (row := 1523) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_10
  simp only [← c93402, ← c93403, ← c93404] at e_1523_10
  have e_1523_11 := arithEq_of_rows h (row := 1523) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_11
  simp only [← c93405, ← c93406, k2, ← c93407] at e_1523_11
  have hr := e_1523_11
  simp only [e_1523_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14721)) := by
  have c93408 := (publicBatchWrapper89_copies2919 a h).1
  have c93409 := (publicBatchWrapper89_copies2919 a h).2.1
  have c93410 := (publicBatchWrapper89_copies2919 a h).2.2.1
  have c93411 := (publicBatchWrapper89_copies2919 a h).2.2.2.1
  have c93412 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.1
  have c93413 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_12 := arithEq_of_rows h (row := 1523) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_12
  simp only [← c93408, ← c93409, ← c93410] at e_1523_12
  have e_1523_13 := arithEq_of_rows h (row := 1523) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_13
  simp only [← c93411, ← c93412, k2, ← c93413] at e_1523_13
  have hr := e_1523_13
  simp only [e_1523_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14722)) := by
  have c93414 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.1
  have c93415 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.1
  have c93416 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.1
  have c93417 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.1
  have c93418 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93419 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_14 := arithEq_of_rows h (row := 1523) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_14
  simp only [← c93414, ← c93415, ← c93416] at e_1523_14
  have e_1523_15 := arithEq_of_rows h (row := 1523) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_15
  simp only [← c93417, ← c93418, k2, ← c93419] at e_1523_15
  have hr := e_1523_15
  simp only [e_1523_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14723)) := by
  have c93420 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93421 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93422 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93423 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93424 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93425 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_16 := arithEq_of_rows h (row := 1523) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_16
  simp only [← c93420, ← c93421, ← c93422] at e_1523_16
  have e_1523_17 := arithEq_of_rows h (row := 1523) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_17
  simp only [← c93423, ← c93424, k2, ← c93425] at e_1523_17
  have hr := e_1523_17
  simp only [e_1523_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14724)) := by
  have c93426 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93427 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93428 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93429 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93430 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93431 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_18 := arithEq_of_rows h (row := 1523) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_18
  simp only [← c93426, ← c93427, ← c93428] at e_1523_18
  have e_1523_19 := arithEq_of_rows h (row := 1523) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_19
  simp only [← c93429, ← c93430, k2, ← c93431] at e_1523_19
  have hr := e_1523_19
  simp only [e_1523_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14725)) := by
  have c93432 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93433 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93434 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93435 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93436 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93437 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_0 := arithEq_of_rows h (row := 1524) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_0
  simp only [← c93432, ← c93433, ← c93434] at e_1524_0
  have e_1524_1 := arithEq_of_rows h (row := 1524) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_1
  simp only [← c93435, ← c93436, k2, ← c93437] at e_1524_1
  have hr := e_1524_1
  simp only [e_1524_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14726)) := by
  have c93438 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93439 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93440 := (publicBatchWrapper89_copies2920 a h).1
  have c93441 := (publicBatchWrapper89_copies2920 a h).2.1
  have c93442 := (publicBatchWrapper89_copies2920 a h).2.2.1
  have c93443 := (publicBatchWrapper89_copies2920 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_2 := arithEq_of_rows h (row := 1524) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_2
  simp only [← c93438, ← c93439, ← c93440] at e_1524_2
  have e_1524_3 := arithEq_of_rows h (row := 1524) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_3
  simp only [← c93441, ← c93442, k2, ← c93443] at e_1524_3
  have hr := e_1524_3
  simp only [e_1524_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14727)) := by
  have c93444 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.1
  have c93445 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.1
  have c93446 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.1
  have c93447 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.1
  have c93448 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.1
  have c93449 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_4 := arithEq_of_rows h (row := 1524) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_4
  simp only [← c93444, ← c93445, ← c93446] at e_1524_4
  have e_1524_5 := arithEq_of_rows h (row := 1524) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_5
  simp only [← c93447, ← c93448, k2, ← c93449] at e_1524_5
  have hr := e_1524_5
  simp only [e_1524_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14728)) := by
  have c93450 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93451 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93452 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93453 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93454 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93455 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_6 := arithEq_of_rows h (row := 1524) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_6
  simp only [← c93450, ← c93451, ← c93452] at e_1524_6
  have e_1524_7 := arithEq_of_rows h (row := 1524) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_7
  simp only [← c93453, ← c93454, k2, ← c93455] at e_1524_7
  have hr := e_1524_7
  simp only [e_1524_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14729)) := by
  have c93456 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93457 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93458 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93459 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93460 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93461 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_8 := arithEq_of_rows h (row := 1524) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_8
  simp only [← c93456, ← c93457, ← c93458] at e_1524_8
  have e_1524_9 := arithEq_of_rows h (row := 1524) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_9
  simp only [← c93459, ← c93460, k2, ← c93461] at e_1524_9
  have hr := e_1524_9
  simp only [e_1524_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14730)) := by
  have c93462 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93463 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93464 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93465 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93466 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93467 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_10 := arithEq_of_rows h (row := 1524) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_10
  simp only [← c93462, ← c93463, ← c93464] at e_1524_10
  have e_1524_11 := arithEq_of_rows h (row := 1524) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_11
  simp only [← c93465, ← c93466, k2, ← c93467] at e_1524_11
  have hr := e_1524_11
  simp only [e_1524_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14731)) := by
  have c93468 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93469 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93470 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93471 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93472 := (publicBatchWrapper89_copies2921 a h).1
  have c93473 := (publicBatchWrapper89_copies2921 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_12 := arithEq_of_rows h (row := 1524) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_12
  simp only [← c93468, ← c93469, ← c93470] at e_1524_12
  have e_1524_13 := arithEq_of_rows h (row := 1524) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_13
  simp only [← c93471, ← c93472, k2, ← c93473] at e_1524_13
  have hr := e_1524_13
  simp only [e_1524_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14732)) := by
  have c93474 := (publicBatchWrapper89_copies2921 a h).2.2.1
  have c93475 := (publicBatchWrapper89_copies2921 a h).2.2.2.1
  have c93476 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.1
  have c93477 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.1
  have c93478 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.1
  have c93479 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_14 := arithEq_of_rows h (row := 1524) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_14
  simp only [← c93474, ← c93475, ← c93476] at e_1524_14
  have e_1524_15 := arithEq_of_rows h (row := 1524) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_15
  simp only [← c93477, ← c93478, k2, ← c93479] at e_1524_15
  have hr := e_1524_15
  simp only [e_1524_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14733)) := by
  have c93480 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.1
  have c93481 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.1
  have c93482 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93483 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93484 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93485 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_16 := arithEq_of_rows h (row := 1524) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_16
  simp only [← c93480, ← c93481, ← c93482] at e_1524_16
  have e_1524_17 := arithEq_of_rows h (row := 1524) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_17
  simp only [← c93483, ← c93484, k2, ← c93485] at e_1524_17
  have hr := e_1524_17
  simp only [e_1524_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14734)) := by
  have c93486 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93487 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93488 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93489 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93490 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93491 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_18 := arithEq_of_rows h (row := 1524) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_18
  simp only [← c93486, ← c93487, ← c93488] at e_1524_18
  have e_1524_19 := arithEq_of_rows h (row := 1524) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_19
  simp only [← c93489, ← c93490, k2, ← c93491] at e_1524_19
  have hr := e_1524_19
  simp only [e_1524_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14735)) := by
  have c93492 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93493 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93494 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93495 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93496 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93497 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_0 := arithEq_of_rows h (row := 1525) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_0
  simp only [← c93492, ← c93493, ← c93494] at e_1525_0
  have e_1525_1 := arithEq_of_rows h (row := 1525) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_1
  simp only [← c93495, ← c93496, k2, ← c93497] at e_1525_1
  have hr := e_1525_1
  simp only [e_1525_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14736)) := by
  have c93498 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93499 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93500 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93501 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93502 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93503 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_2 := arithEq_of_rows h (row := 1525) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_2
  simp only [← c93498, ← c93499, ← c93500] at e_1525_2
  have e_1525_3 := arithEq_of_rows h (row := 1525) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_3
  simp only [← c93501, ← c93502, k2, ← c93503] at e_1525_3
  have hr := e_1525_3
  simp only [e_1525_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14737)) := by
  have c93504 := (publicBatchWrapper89_copies2922 a h).1
  have c93505 := (publicBatchWrapper89_copies2922 a h).2.1
  have c93506 := (publicBatchWrapper89_copies2922 a h).2.2.1
  have c93507 := (publicBatchWrapper89_copies2922 a h).2.2.2.1
  have c93508 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.1
  have c93509 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_4 := arithEq_of_rows h (row := 1525) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_4
  simp only [← c93504, ← c93505, ← c93506] at e_1525_4
  have e_1525_5 := arithEq_of_rows h (row := 1525) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_5
  simp only [← c93507, ← c93508, k2, ← c93509] at e_1525_5
  have hr := e_1525_5
  simp only [e_1525_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14738)) := by
  have c93510 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.1
  have c93511 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.1
  have c93512 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.1
  have c93513 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.1
  have c93514 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93515 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_6 := arithEq_of_rows h (row := 1525) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_6
  simp only [← c93510, ← c93511, ← c93512] at e_1525_6
  have e_1525_7 := arithEq_of_rows h (row := 1525) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_7
  simp only [← c93513, ← c93514, k2, ← c93515] at e_1525_7
  have hr := e_1525_7
  simp only [e_1525_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14739)) := by
  have c93516 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93517 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93518 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93519 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93520 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93521 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_8 := arithEq_of_rows h (row := 1525) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_8
  simp only [← c93516, ← c93517, ← c93518] at e_1525_8
  have e_1525_9 := arithEq_of_rows h (row := 1525) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_9
  simp only [← c93519, ← c93520, k2, ← c93521] at e_1525_9
  have hr := e_1525_9
  simp only [e_1525_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14740)) := by
  have c93522 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93523 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93524 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93525 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93526 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93527 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_10 := arithEq_of_rows h (row := 1525) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_10
  simp only [← c93522, ← c93523, ← c93524] at e_1525_10
  have e_1525_11 := arithEq_of_rows h (row := 1525) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_11
  simp only [← c93525, ← c93526, k2, ← c93527] at e_1525_11
  have hr := e_1525_11
  simp only [e_1525_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14741)) := by
  have c93528 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93529 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93530 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93531 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93532 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93533 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_12 := arithEq_of_rows h (row := 1525) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_12
  simp only [← c93528, ← c93529, ← c93530] at e_1525_12
  have e_1525_13 := arithEq_of_rows h (row := 1525) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_13
  simp only [← c93531, ← c93532, k2, ← c93533] at e_1525_13
  have hr := e_1525_13
  simp only [e_1525_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14742)) := by
  have c93534 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93535 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93536 := (publicBatchWrapper89_copies2923 a h).1
  have c93537 := (publicBatchWrapper89_copies2923 a h).2.1
  have c93538 := (publicBatchWrapper89_copies2923 a h).2.2.1
  have c93539 := (publicBatchWrapper89_copies2923 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_14 := arithEq_of_rows h (row := 1525) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_14
  simp only [← c93534, ← c93535, ← c93536] at e_1525_14
  have e_1525_15 := arithEq_of_rows h (row := 1525) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_15
  simp only [← c93537, ← c93538, k2, ← c93539] at e_1525_15
  have hr := e_1525_15
  simp only [e_1525_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14743)) := by
  have c93540 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.1
  have c93541 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.1
  have c93542 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.1
  have c93543 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.1
  have c93544 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.1
  have c93545 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_16 := arithEq_of_rows h (row := 1525) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_16
  simp only [← c93540, ← c93541, ← c93542] at e_1525_16
  have e_1525_17 := arithEq_of_rows h (row := 1525) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_17
  simp only [← c93543, ← c93544, k2, ← c93545] at e_1525_17
  have hr := e_1525_17
  simp only [e_1525_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14744)) := by
  have c93546 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93547 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93548 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93549 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93550 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93551 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_18 := arithEq_of_rows h (row := 1525) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_18
  simp only [← c93546, ← c93547, ← c93548] at e_1525_18
  have e_1525_19 := arithEq_of_rows h (row := 1525) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_19
  simp only [← c93549, ← c93550, k2, ← c93551] at e_1525_19
  have hr := e_1525_19
  simp only [e_1525_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14745)) := by
  have c93552 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93553 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93554 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93555 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93556 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93557 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_0 := arithEq_of_rows h (row := 1526) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_0
  simp only [← c93552, ← c93553, ← c93554] at e_1526_0
  have e_1526_1 := arithEq_of_rows h (row := 1526) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_1
  simp only [← c93555, ← c93556, k2, ← c93557] at e_1526_1
  have hr := e_1526_1
  simp only [e_1526_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14746)) := by
  have c93558 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93559 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93560 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93561 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93562 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93563 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_2 := arithEq_of_rows h (row := 1526) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_2
  simp only [← c93558, ← c93559, ← c93560] at e_1526_2
  have e_1526_3 := arithEq_of_rows h (row := 1526) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_3
  simp only [← c93561, ← c93562, k2, ← c93563] at e_1526_3
  have hr := e_1526_3
  simp only [e_1526_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14747)) := by
  have c93564 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93565 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93566 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93567 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93568 := (publicBatchWrapper89_copies2924 a h).1
  have c93569 := (publicBatchWrapper89_copies2924 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_4 := arithEq_of_rows h (row := 1526) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_4
  simp only [← c93564, ← c93565, ← c93566] at e_1526_4
  have e_1526_5 := arithEq_of_rows h (row := 1526) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_5
  simp only [← c93567, ← c93568, k2, ← c93569] at e_1526_5
  have hr := e_1526_5
  simp only [e_1526_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14748)) := by
  have c93570 := (publicBatchWrapper89_copies2924 a h).2.2.1
  have c93571 := (publicBatchWrapper89_copies2924 a h).2.2.2.1
  have c93572 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.1
  have c93573 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.1
  have c93574 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.1
  have c93575 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_6 := arithEq_of_rows h (row := 1526) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_6
  simp only [← c93570, ← c93571, ← c93572] at e_1526_6
  have e_1526_7 := arithEq_of_rows h (row := 1526) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_7
  simp only [← c93573, ← c93574, k2, ← c93575] at e_1526_7
  have hr := e_1526_7
  simp only [e_1526_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14749)) := by
  have c93576 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.1
  have c93577 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.1
  have c93578 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93579 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93580 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93581 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_8 := arithEq_of_rows h (row := 1526) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_8
  simp only [← c93576, ← c93577, ← c93578] at e_1526_8
  have e_1526_9 := arithEq_of_rows h (row := 1526) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_9
  simp only [← c93579, ← c93580, k2, ← c93581] at e_1526_9
  have hr := e_1526_9
  simp only [e_1526_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14750)) := by
  have c93582 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93583 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93584 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93585 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93586 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93587 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_10 := arithEq_of_rows h (row := 1526) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_10
  simp only [← c93582, ← c93583, ← c93584] at e_1526_10
  have e_1526_11 := arithEq_of_rows h (row := 1526) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_11
  simp only [← c93585, ← c93586, k2, ← c93587] at e_1526_11
  have hr := e_1526_11
  simp only [e_1526_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14751)) := by
  have c93588 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93589 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93590 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93591 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93592 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93593 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_12 := arithEq_of_rows h (row := 1526) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_12
  simp only [← c93588, ← c93589, ← c93590] at e_1526_12
  have e_1526_13 := arithEq_of_rows h (row := 1526) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_13
  simp only [← c93591, ← c93592, k2, ← c93593] at e_1526_13
  have hr := e_1526_13
  simp only [e_1526_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14752)) := by
  have c93594 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93595 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93596 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93597 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93598 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93599 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_14 := arithEq_of_rows h (row := 1526) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_14
  simp only [← c93594, ← c93595, ← c93596] at e_1526_14
  have e_1526_15 := arithEq_of_rows h (row := 1526) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_15
  simp only [← c93597, ← c93598, k2, ← c93599] at e_1526_15
  have hr := e_1526_15
  simp only [e_1526_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14753)) := by
  have c93600 := (publicBatchWrapper89_copies2925 a h).1
  have c93601 := (publicBatchWrapper89_copies2925 a h).2.1
  have c93602 := (publicBatchWrapper89_copies2925 a h).2.2.1
  have c93603 := (publicBatchWrapper89_copies2925 a h).2.2.2.1
  have c93604 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.1
  have c93605 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_16 := arithEq_of_rows h (row := 1526) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_16
  simp only [← c93600, ← c93601, ← c93602] at e_1526_16
  have e_1526_17 := arithEq_of_rows h (row := 1526) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_17
  simp only [← c93603, ← c93604, k2, ← c93605] at e_1526_17
  have hr := e_1526_17
  simp only [e_1526_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14754)) := by
  have c93606 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.1
  have c93607 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.1
  have c93608 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.1
  have c93609 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.1
  have c93610 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93611 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_18 := arithEq_of_rows h (row := 1526) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_18
  simp only [← c93606, ← c93607, ← c93608] at e_1526_18
  have e_1526_19 := arithEq_of_rows h (row := 1526) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_19
  simp only [← c93609, ← c93610, k2, ← c93611] at e_1526_19
  have hr := e_1526_19
  simp only [e_1526_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14755)) := by
  have c93612 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93613 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93614 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93615 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93616 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93617 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_0 := arithEq_of_rows h (row := 1527) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_0
  simp only [← c93612, ← c93613, ← c93614] at e_1527_0
  have e_1527_1 := arithEq_of_rows h (row := 1527) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_1
  simp only [← c93615, ← c93616, k2, ← c93617] at e_1527_1
  have hr := e_1527_1
  simp only [e_1527_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14756)) := by
  have c93618 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93619 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93620 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93621 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93622 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93623 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_2 := arithEq_of_rows h (row := 1527) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_2
  simp only [← c93618, ← c93619, ← c93620] at e_1527_2
  have e_1527_3 := arithEq_of_rows h (row := 1527) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_3
  simp only [← c93621, ← c93622, k2, ← c93623] at e_1527_3
  have hr := e_1527_3
  simp only [e_1527_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14757)) := by
  have c93624 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93625 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93626 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93627 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93628 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93629 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_4 := arithEq_of_rows h (row := 1527) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_4
  simp only [← c93624, ← c93625, ← c93626] at e_1527_4
  have e_1527_5 := arithEq_of_rows h (row := 1527) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_5
  simp only [← c93627, ← c93628, k2, ← c93629] at e_1527_5
  have hr := e_1527_5
  simp only [e_1527_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14758)) := by
  have c93630 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93631 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93632 := (publicBatchWrapper89_copies2926 a h).1
  have c93633 := (publicBatchWrapper89_copies2926 a h).2.1
  have c93634 := (publicBatchWrapper89_copies2926 a h).2.2.1
  have c93635 := (publicBatchWrapper89_copies2926 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_6 := arithEq_of_rows h (row := 1527) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_6
  simp only [← c93630, ← c93631, ← c93632] at e_1527_6
  have e_1527_7 := arithEq_of_rows h (row := 1527) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_7
  simp only [← c93633, ← c93634, k2, ← c93635] at e_1527_7
  have hr := e_1527_7
  simp only [e_1527_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14759)) := by
  have c93636 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.1
  have c93637 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.1
  have c93638 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.1
  have c93639 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.1
  have c93640 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.1
  have c93641 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_8 := arithEq_of_rows h (row := 1527) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_8
  simp only [← c93636, ← c93637, ← c93638] at e_1527_8
  have e_1527_9 := arithEq_of_rows h (row := 1527) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_9
  simp only [← c93639, ← c93640, k2, ← c93641] at e_1527_9
  have hr := e_1527_9
  simp only [e_1527_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14760)) := by
  have c93642 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93643 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93644 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93645 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93646 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93647 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_10 := arithEq_of_rows h (row := 1527) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_10
  simp only [← c93642, ← c93643, ← c93644] at e_1527_10
  have e_1527_11 := arithEq_of_rows h (row := 1527) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_11
  simp only [← c93645, ← c93646, k2, ← c93647] at e_1527_11
  have hr := e_1527_11
  simp only [e_1527_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14761)) := by
  have c93648 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93649 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93650 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93651 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93652 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93653 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_12 := arithEq_of_rows h (row := 1527) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_12
  simp only [← c93648, ← c93649, ← c93650] at e_1527_12
  have e_1527_13 := arithEq_of_rows h (row := 1527) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_13
  simp only [← c93651, ← c93652, k2, ← c93653] at e_1527_13
  have hr := e_1527_13
  simp only [e_1527_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14762)) := by
  have c93654 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93655 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93656 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93657 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93658 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93659 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_14 := arithEq_of_rows h (row := 1527) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_14
  simp only [← c93654, ← c93655, ← c93656] at e_1527_14
  have e_1527_15 := arithEq_of_rows h (row := 1527) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_15
  simp only [← c93657, ← c93658, k2, ← c93659] at e_1527_15
  have hr := e_1527_15
  simp only [e_1527_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14763)) := by
  have c93660 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93661 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93662 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93663 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93664 := (publicBatchWrapper89_copies2927 a h).1
  have c93665 := (publicBatchWrapper89_copies2927 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_16 := arithEq_of_rows h (row := 1527) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_16
  simp only [← c93660, ← c93661, ← c93662] at e_1527_16
  have e_1527_17 := arithEq_of_rows h (row := 1527) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_17
  simp only [← c93663, ← c93664, k2, ← c93665] at e_1527_17
  have hr := e_1527_17
  simp only [e_1527_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14764)) := by
  have c93666 := (publicBatchWrapper89_copies2927 a h).2.2.1
  have c93667 := (publicBatchWrapper89_copies2927 a h).2.2.2.1
  have c93668 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.1
  have c93669 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.1
  have c93670 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.1
  have c93671 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_18 := arithEq_of_rows h (row := 1527) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_18
  simp only [← c93666, ← c93667, ← c93668] at e_1527_18
  have e_1527_19 := arithEq_of_rows h (row := 1527) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_19
  simp only [← c93669, ← c93670, k2, ← c93671] at e_1527_19
  have hr := e_1527_19
  simp only [e_1527_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14765)) := by
  have c93672 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.1
  have c93673 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.1
  have c93674 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93675 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93676 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93677 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_0 := arithEq_of_rows h (row := 1528) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_0
  simp only [← c93672, ← c93673, ← c93674] at e_1528_0
  have e_1528_1 := arithEq_of_rows h (row := 1528) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_1
  simp only [← c93675, ← c93676, k2, ← c93677] at e_1528_1
  have hr := e_1528_1
  simp only [e_1528_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14766)) := by
  have c93678 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93679 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93680 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93681 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93682 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93683 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_2 := arithEq_of_rows h (row := 1528) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_2
  simp only [← c93678, ← c93679, ← c93680] at e_1528_2
  have e_1528_3 := arithEq_of_rows h (row := 1528) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_3
  simp only [← c93681, ← c93682, k2, ← c93683] at e_1528_3
  have hr := e_1528_3
  simp only [e_1528_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14767)) := by
  have c93684 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93685 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93686 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93687 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93688 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93689 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_4 := arithEq_of_rows h (row := 1528) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_4
  simp only [← c93684, ← c93685, ← c93686] at e_1528_4
  have e_1528_5 := arithEq_of_rows h (row := 1528) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_5
  simp only [← c93687, ← c93688, k2, ← c93689] at e_1528_5
  have hr := e_1528_5
  simp only [e_1528_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14768)) := by
  have c93690 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93691 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93692 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93693 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93694 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93695 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_6 := arithEq_of_rows h (row := 1528) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_6
  simp only [← c93690, ← c93691, ← c93692] at e_1528_6
  have e_1528_7 := arithEq_of_rows h (row := 1528) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_7
  simp only [← c93693, ← c93694, k2, ← c93695] at e_1528_7
  have hr := e_1528_7
  simp only [e_1528_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14769)) := by
  have c93696 := (publicBatchWrapper89_copies2928 a h).1
  have c93697 := (publicBatchWrapper89_copies2928 a h).2.1
  have c93698 := (publicBatchWrapper89_copies2928 a h).2.2.1
  have c93699 := (publicBatchWrapper89_copies2928 a h).2.2.2.1
  have c93700 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.1
  have c93701 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_8 := arithEq_of_rows h (row := 1528) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_8
  simp only [← c93696, ← c93697, ← c93698] at e_1528_8
  have e_1528_9 := arithEq_of_rows h (row := 1528) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_9
  simp only [← c93699, ← c93700, k2, ← c93701] at e_1528_9
  have hr := e_1528_9
  simp only [e_1528_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14810)) := by
  have c93702 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.1
  have c93703 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.1
  have c93704 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.1
  have c93705 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.1
  have c93706 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93707 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_10 := arithEq_of_rows h (row := 1528) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_10
  simp only [← c93702, ← c93703, ← c93704] at e_1528_10
  have e_1528_11 := arithEq_of_rows h (row := 1528) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_11
  simp only [← c93705, ← c93706, k2, ← c93707] at e_1528_11
  have hr := e_1528_11
  simp only [e_1528_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14811)) := by
  have c93708 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93709 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93710 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93711 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93712 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93713 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_12 := arithEq_of_rows h (row := 1528) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_12
  simp only [← c93708, ← c93709, ← c93710] at e_1528_12
  have e_1528_13 := arithEq_of_rows h (row := 1528) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_13
  simp only [← c93711, ← c93712, k2, ← c93713] at e_1528_13
  have hr := e_1528_13
  simp only [e_1528_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14812)) := by
  have c93714 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93715 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93716 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93717 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93718 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93719 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_14 := arithEq_of_rows h (row := 1528) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_14
  simp only [← c93714, ← c93715, ← c93716] at e_1528_14
  have e_1528_15 := arithEq_of_rows h (row := 1528) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_15
  simp only [← c93717, ← c93718, k2, ← c93719] at e_1528_15
  have hr := e_1528_15
  simp only [e_1528_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14813)) := by
  have c93720 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93721 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93722 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93723 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93724 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93725 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_16 := arithEq_of_rows h (row := 1528) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_16
  simp only [← c93720, ← c93721, ← c93722] at e_1528_16
  have e_1528_17 := arithEq_of_rows h (row := 1528) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_17
  simp only [← c93723, ← c93724, k2, ← c93725] at e_1528_17
  have hr := e_1528_17
  simp only [e_1528_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14814)) := by
  have c93726 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93727 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93728 := (publicBatchWrapper89_copies2929 a h).1
  have c93729 := (publicBatchWrapper89_copies2929 a h).2.1
  have c93730 := (publicBatchWrapper89_copies2929 a h).2.2.1
  have c93731 := (publicBatchWrapper89_copies2929 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_18 := arithEq_of_rows h (row := 1528) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_18
  simp only [← c93726, ← c93727, ← c93728] at e_1528_18
  have e_1528_19 := arithEq_of_rows h (row := 1528) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_19
  simp only [← c93729, ← c93730, k2, ← c93731] at e_1528_19
  have hr := e_1528_19
  simp only [e_1528_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14815)) := by
  have c93732 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.1
  have c93733 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.1
  have c93734 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.1
  have c93735 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.1
  have c93736 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.1
  have c93737 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_0 := arithEq_of_rows h (row := 1529) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_0
  simp only [← c93732, ← c93733, ← c93734] at e_1529_0
  have e_1529_1 := arithEq_of_rows h (row := 1529) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_1
  simp only [← c93735, ← c93736, k2, ← c93737] at e_1529_1
  have hr := e_1529_1
  simp only [e_1529_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14816)) := by
  have c93738 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93739 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93740 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93741 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93742 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93743 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_2 := arithEq_of_rows h (row := 1529) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_2
  simp only [← c93738, ← c93739, ← c93740] at e_1529_2
  have e_1529_3 := arithEq_of_rows h (row := 1529) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_3
  simp only [← c93741, ← c93742, k2, ← c93743] at e_1529_3
  have hr := e_1529_3
  simp only [e_1529_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14817)) := by
  have c93744 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93745 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93746 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93747 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93748 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93749 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_4 := arithEq_of_rows h (row := 1529) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_4
  simp only [← c93744, ← c93745, ← c93746] at e_1529_4
  have e_1529_5 := arithEq_of_rows h (row := 1529) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_5
  simp only [← c93747, ← c93748, k2, ← c93749] at e_1529_5
  have hr := e_1529_5
  simp only [e_1529_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14818)) := by
  have c93750 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93751 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93752 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93753 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93754 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93755 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_6 := arithEq_of_rows h (row := 1529) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_6
  simp only [← c93750, ← c93751, ← c93752] at e_1529_6
  have e_1529_7 := arithEq_of_rows h (row := 1529) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_7
  simp only [← c93753, ← c93754, k2, ← c93755] at e_1529_7
  have hr := e_1529_7
  simp only [e_1529_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14819)) := by
  have c93756 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93757 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93758 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93759 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93760 := (publicBatchWrapper89_copies2930 a h).1
  have c93761 := (publicBatchWrapper89_copies2930 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_8 := arithEq_of_rows h (row := 1529) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_8
  simp only [← c93756, ← c93757, ← c93758] at e_1529_8
  have e_1529_9 := arithEq_of_rows h (row := 1529) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_9
  simp only [← c93759, ← c93760, k2, ← c93761] at e_1529_9
  have hr := e_1529_9
  simp only [e_1529_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14820)) := by
  have c93762 := (publicBatchWrapper89_copies2930 a h).2.2.1
  have c93763 := (publicBatchWrapper89_copies2930 a h).2.2.2.1
  have c93764 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.1
  have c93765 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.1
  have c93766 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.1
  have c93767 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_10 := arithEq_of_rows h (row := 1529) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_10
  simp only [← c93762, ← c93763, ← c93764] at e_1529_10
  have e_1529_11 := arithEq_of_rows h (row := 1529) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_11
  simp only [← c93765, ← c93766, k2, ← c93767] at e_1529_11
  have hr := e_1529_11
  simp only [e_1529_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14821)) := by
  have c93768 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.1
  have c93769 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.1
  have c93770 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93771 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93772 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93773 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_12 := arithEq_of_rows h (row := 1529) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_12
  simp only [← c93768, ← c93769, ← c93770] at e_1529_12
  have e_1529_13 := arithEq_of_rows h (row := 1529) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_13
  simp only [← c93771, ← c93772, k2, ← c93773] at e_1529_13
  have hr := e_1529_13
  simp only [e_1529_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14822)) := by
  have c93774 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93775 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93776 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93777 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93778 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93779 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_14 := arithEq_of_rows h (row := 1529) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_14
  simp only [← c93774, ← c93775, ← c93776] at e_1529_14
  have e_1529_15 := arithEq_of_rows h (row := 1529) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_15
  simp only [← c93777, ← c93778, k2, ← c93779] at e_1529_15
  have hr := e_1529_15
  simp only [e_1529_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14823)) := by
  have c93780 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93781 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93782 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93783 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93784 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93785 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_16 := arithEq_of_rows h (row := 1529) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_16
  simp only [← c93780, ← c93781, ← c93782] at e_1529_16
  have e_1529_17 := arithEq_of_rows h (row := 1529) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_17
  simp only [← c93783, ← c93784, k2, ← c93785] at e_1529_17
  have hr := e_1529_17
  simp only [e_1529_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14824)) := by
  have c93786 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93787 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93788 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93789 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93790 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93791 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_18 := arithEq_of_rows h (row := 1529) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_18
  simp only [← c93786, ← c93787, ← c93788] at e_1529_18
  have e_1529_19 := arithEq_of_rows h (row := 1529) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_19
  simp only [← c93789, ← c93790, k2, ← c93791] at e_1529_19
  have hr := e_1529_19
  simp only [e_1529_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14825)) := by
  have c93792 := (publicBatchWrapper89_copies2931 a h).1
  have c93793 := (publicBatchWrapper89_copies2931 a h).2.1
  have c93794 := (publicBatchWrapper89_copies2931 a h).2.2.1
  have c93795 := (publicBatchWrapper89_copies2931 a h).2.2.2.1
  have c93796 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.1
  have c93797 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_0 := arithEq_of_rows h (row := 1530) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_0
  simp only [← c93792, ← c93793, ← c93794] at e_1530_0
  have e_1530_1 := arithEq_of_rows h (row := 1530) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_1
  simp only [← c93795, ← c93796, k2, ← c93797] at e_1530_1
  have hr := e_1530_1
  simp only [e_1530_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14826)) := by
  have c93798 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.1
  have c93799 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.1
  have c93800 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.1
  have c93801 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.1
  have c93802 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93803 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_2 := arithEq_of_rows h (row := 1530) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_2
  simp only [← c93798, ← c93799, ← c93800] at e_1530_2
  have e_1530_3 := arithEq_of_rows h (row := 1530) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_3
  simp only [← c93801, ← c93802, k2, ← c93803] at e_1530_3
  have hr := e_1530_3
  simp only [e_1530_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14827)) := by
  have c93804 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93805 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93806 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93807 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93808 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93809 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_4 := arithEq_of_rows h (row := 1530) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_4
  simp only [← c93804, ← c93805, ← c93806] at e_1530_4
  have e_1530_5 := arithEq_of_rows h (row := 1530) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_5
  simp only [← c93807, ← c93808, k2, ← c93809] at e_1530_5
  have hr := e_1530_5
  simp only [e_1530_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14828)) := by
  have c93810 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93811 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93812 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93813 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93814 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93815 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_6 := arithEq_of_rows h (row := 1530) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_6
  simp only [← c93810, ← c93811, ← c93812] at e_1530_6
  have e_1530_7 := arithEq_of_rows h (row := 1530) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_7
  simp only [← c93813, ← c93814, k2, ← c93815] at e_1530_7
  have hr := e_1530_7
  simp only [e_1530_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14829)) := by
  have c93816 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93817 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93818 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93819 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93820 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93821 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_8 := arithEq_of_rows h (row := 1530) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_8
  simp only [← c93816, ← c93817, ← c93818] at e_1530_8
  have e_1530_9 := arithEq_of_rows h (row := 1530) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_9
  simp only [← c93819, ← c93820, k2, ← c93821] at e_1530_9
  have hr := e_1530_9
  simp only [e_1530_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14830)) := by
  have c93822 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93823 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93824 := (publicBatchWrapper89_copies2932 a h).1
  have c93825 := (publicBatchWrapper89_copies2932 a h).2.1
  have c93826 := (publicBatchWrapper89_copies2932 a h).2.2.1
  have c93827 := (publicBatchWrapper89_copies2932 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_10 := arithEq_of_rows h (row := 1530) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_10
  simp only [← c93822, ← c93823, ← c93824] at e_1530_10
  have e_1530_11 := arithEq_of_rows h (row := 1530) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_11
  simp only [← c93825, ← c93826, k2, ← c93827] at e_1530_11
  have hr := e_1530_11
  simp only [e_1530_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14831)) := by
  have c93828 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.1
  have c93829 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.1
  have c93830 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.1
  have c93831 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.1
  have c93832 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.1
  have c93833 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_12 := arithEq_of_rows h (row := 1530) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_12
  simp only [← c93828, ← c93829, ← c93830] at e_1530_12
  have e_1530_13 := arithEq_of_rows h (row := 1530) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_13
  simp only [← c93831, ← c93832, k2, ← c93833] at e_1530_13
  have hr := e_1530_13
  simp only [e_1530_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14832)) := by
  have c93834 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93835 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93836 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93837 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93838 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93839 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_14 := arithEq_of_rows h (row := 1530) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_14
  simp only [← c93834, ← c93835, ← c93836] at e_1530_14
  have e_1530_15 := arithEq_of_rows h (row := 1530) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_15
  simp only [← c93837, ← c93838, k2, ← c93839] at e_1530_15
  have hr := e_1530_15
  simp only [e_1530_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14833)) := by
  have c93840 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93841 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93842 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93843 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93844 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93845 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_16 := arithEq_of_rows h (row := 1530) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_16
  simp only [← c93840, ← c93841, ← c93842] at e_1530_16
  have e_1530_17 := arithEq_of_rows h (row := 1530) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_17
  simp only [← c93843, ← c93844, k2, ← c93845] at e_1530_17
  have hr := e_1530_17
  simp only [e_1530_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14834)) := by
  have c93846 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93847 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93848 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93849 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93850 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93851 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_18 := arithEq_of_rows h (row := 1530) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_18
  simp only [← c93846, ← c93847, ← c93848] at e_1530_18
  have e_1530_19 := arithEq_of_rows h (row := 1530) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_19
  simp only [← c93849, ← c93850, k2, ← c93851] at e_1530_19
  have hr := e_1530_19
  simp only [e_1530_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14835)) := by
  have c93852 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93853 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93854 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93855 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93856 := (publicBatchWrapper89_copies2933 a h).1
  have c93857 := (publicBatchWrapper89_copies2933 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_0 := arithEq_of_rows h (row := 1531) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_0
  simp only [← c93852, ← c93853, ← c93854] at e_1531_0
  have e_1531_1 := arithEq_of_rows h (row := 1531) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_1
  simp only [← c93855, ← c93856, k2, ← c93857] at e_1531_1
  have hr := e_1531_1
  simp only [e_1531_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14836)) := by
  have c93858 := (publicBatchWrapper89_copies2933 a h).2.2.1
  have c93859 := (publicBatchWrapper89_copies2933 a h).2.2.2.1
  have c93860 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.1
  have c93861 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.1
  have c93862 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.1
  have c93863 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_2 := arithEq_of_rows h (row := 1531) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_2
  simp only [← c93858, ← c93859, ← c93860] at e_1531_2
  have e_1531_3 := arithEq_of_rows h (row := 1531) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_3
  simp only [← c93861, ← c93862, k2, ← c93863] at e_1531_3
  have hr := e_1531_3
  simp only [e_1531_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14837)) := by
  have c93864 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.1
  have c93865 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.1
  have c93866 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93867 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93868 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93869 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_4 := arithEq_of_rows h (row := 1531) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_4
  simp only [← c93864, ← c93865, ← c93866] at e_1531_4
  have e_1531_5 := arithEq_of_rows h (row := 1531) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_5
  simp only [← c93867, ← c93868, k2, ← c93869] at e_1531_5
  have hr := e_1531_5
  simp only [e_1531_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14838)) := by
  have c93870 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93871 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93872 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93873 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93874 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93875 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_6 := arithEq_of_rows h (row := 1531) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_6
  simp only [← c93870, ← c93871, ← c93872] at e_1531_6
  have e_1531_7 := arithEq_of_rows h (row := 1531) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_7
  simp only [← c93873, ← c93874, k2, ← c93875] at e_1531_7
  have hr := e_1531_7
  simp only [e_1531_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14839)) := by
  have c93876 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93877 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93878 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93879 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93880 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93881 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_8 := arithEq_of_rows h (row := 1531) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_8
  simp only [← c93876, ← c93877, ← c93878] at e_1531_8
  have e_1531_9 := arithEq_of_rows h (row := 1531) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_9
  simp only [← c93879, ← c93880, k2, ← c93881] at e_1531_9
  have hr := e_1531_9
  simp only [e_1531_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14840)) := by
  have c93882 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93883 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93884 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93885 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93886 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93887 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_10 := arithEq_of_rows h (row := 1531) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_10
  simp only [← c93882, ← c93883, ← c93884] at e_1531_10
  have e_1531_11 := arithEq_of_rows h (row := 1531) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_11
  simp only [← c93885, ← c93886, k2, ← c93887] at e_1531_11
  have hr := e_1531_11
  simp only [e_1531_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14841)) := by
  have c93888 := (publicBatchWrapper89_copies2934 a h).1
  have c93889 := (publicBatchWrapper89_copies2934 a h).2.1
  have c93890 := (publicBatchWrapper89_copies2934 a h).2.2.1
  have c93891 := (publicBatchWrapper89_copies2934 a h).2.2.2.1
  have c93892 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.1
  have c93893 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_12 := arithEq_of_rows h (row := 1531) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_12
  simp only [← c93888, ← c93889, ← c93890] at e_1531_12
  have e_1531_13 := arithEq_of_rows h (row := 1531) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_13
  simp only [← c93891, ← c93892, k2, ← c93893] at e_1531_13
  have hr := e_1531_13
  simp only [e_1531_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14842)) := by
  have c93894 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.1
  have c93895 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.1
  have c93896 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.1
  have c93897 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.1
  have c93898 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93899 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_14 := arithEq_of_rows h (row := 1531) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_14
  simp only [← c93894, ← c93895, ← c93896] at e_1531_14
  have e_1531_15 := arithEq_of_rows h (row := 1531) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_15
  simp only [← c93897, ← c93898, k2, ← c93899] at e_1531_15
  have hr := e_1531_15
  simp only [e_1531_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14843)) := by
  have c93900 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93901 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93902 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93903 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93904 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93905 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_16 := arithEq_of_rows h (row := 1531) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_16
  simp only [← c93900, ← c93901, ← c93902] at e_1531_16
  have e_1531_17 := arithEq_of_rows h (row := 1531) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_17
  simp only [← c93903, ← c93904, k2, ← c93905] at e_1531_17
  have hr := e_1531_17
  simp only [e_1531_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14844)) := by
  have c93906 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93907 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93908 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93909 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93910 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93911 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_18 := arithEq_of_rows h (row := 1531) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_18
  simp only [← c93906, ← c93907, ← c93908] at e_1531_18
  have e_1531_19 := arithEq_of_rows h (row := 1531) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_19
  simp only [← c93909, ← c93910, k2, ← c93911] at e_1531_19
  have hr := e_1531_19
  simp only [e_1531_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14845)) := by
  have c93912 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93913 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93914 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93915 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93916 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93917 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_0 := arithEq_of_rows h (row := 1532) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_0
  simp only [← c93912, ← c93913, ← c93914] at e_1532_0
  have e_1532_1 := arithEq_of_rows h (row := 1532) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_1
  simp only [← c93915, ← c93916, k2, ← c93917] at e_1532_1
  have hr := e_1532_1
  simp only [e_1532_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14846)) := by
  have c93918 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93919 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93920 := (publicBatchWrapper89_copies2935 a h).1
  have c93921 := (publicBatchWrapper89_copies2935 a h).2.1
  have c93922 := (publicBatchWrapper89_copies2935 a h).2.2.1
  have c93923 := (publicBatchWrapper89_copies2935 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_2 := arithEq_of_rows h (row := 1532) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_2
  simp only [← c93918, ← c93919, ← c93920] at e_1532_2
  have e_1532_3 := arithEq_of_rows h (row := 1532) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_3
  simp only [← c93921, ← c93922, k2, ← c93923] at e_1532_3
  have hr := e_1532_3
  simp only [e_1532_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14847)) := by
  have c93924 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.1
  have c93925 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.1
  have c93926 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.1
  have c93927 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.1
  have c93928 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.1
  have c93929 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_4 := arithEq_of_rows h (row := 1532) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_4
  simp only [← c93924, ← c93925, ← c93926] at e_1532_4
  have e_1532_5 := arithEq_of_rows h (row := 1532) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_5
  simp only [← c93927, ← c93928, k2, ← c93929] at e_1532_5
  have hr := e_1532_5
  simp only [e_1532_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14848)) := by
  have c93930 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93931 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93932 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93933 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93934 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93935 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_6 := arithEq_of_rows h (row := 1532) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_6
  simp only [← c93930, ← c93931, ← c93932] at e_1532_6
  have e_1532_7 := arithEq_of_rows h (row := 1532) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_7
  simp only [← c93933, ← c93934, k2, ← c93935] at e_1532_7
  have hr := e_1532_7
  simp only [e_1532_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14849)) := by
  have c93936 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93937 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93938 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93939 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93940 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93941 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_8 := arithEq_of_rows h (row := 1532) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_8
  simp only [← c93936, ← c93937, ← c93938] at e_1532_8
  have e_1532_9 := arithEq_of_rows h (row := 1532) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_9
  simp only [← c93939, ← c93940, k2, ← c93941] at e_1532_9
  have hr := e_1532_9
  simp only [e_1532_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14850)) := by
  have c93942 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93943 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93944 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93945 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93946 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93947 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_10 := arithEq_of_rows h (row := 1532) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_10
  simp only [← c93942, ← c93943, ← c93944] at e_1532_10
  have e_1532_11 := arithEq_of_rows h (row := 1532) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_11
  simp only [← c93945, ← c93946, k2, ← c93947] at e_1532_11
  have hr := e_1532_11
  simp only [e_1532_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14851)) := by
  have c93948 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93949 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93950 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93951 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93952 := (publicBatchWrapper89_copies2936 a h).1
  have c93953 := (publicBatchWrapper89_copies2936 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_12 := arithEq_of_rows h (row := 1532) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_12
  simp only [← c93948, ← c93949, ← c93950] at e_1532_12
  have e_1532_13 := arithEq_of_rows h (row := 1532) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_13
  simp only [← c93951, ← c93952, k2, ← c93953] at e_1532_13
  have hr := e_1532_13
  simp only [e_1532_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14852)) := by
  have c93954 := (publicBatchWrapper89_copies2936 a h).2.2.1
  have c93955 := (publicBatchWrapper89_copies2936 a h).2.2.2.1
  have c93956 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.1
  have c93957 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.1
  have c93958 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.1
  have c93959 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_14 := arithEq_of_rows h (row := 1532) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_14
  simp only [← c93954, ← c93955, ← c93956] at e_1532_14
  have e_1532_15 := arithEq_of_rows h (row := 1532) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_15
  simp only [← c93957, ← c93958, k2, ← c93959] at e_1532_15
  have hr := e_1532_15
  simp only [e_1532_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14853)) := by
  have c93960 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.1
  have c93961 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.1
  have c93962 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93963 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93964 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93965 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_16 := arithEq_of_rows h (row := 1532) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_16
  simp only [← c93960, ← c93961, ← c93962] at e_1532_16
  have e_1532_17 := arithEq_of_rows h (row := 1532) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_17
  simp only [← c93963, ← c93964, k2, ← c93965] at e_1532_17
  have hr := e_1532_17
  simp only [e_1532_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14854)) := by
  have c93966 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93967 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93968 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93969 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93970 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93971 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_18 := arithEq_of_rows h (row := 1532) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_18
  simp only [← c93966, ← c93967, ← c93968] at e_1532_18
  have e_1532_19 := arithEq_of_rows h (row := 1532) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_19
  simp only [← c93969, ← c93970, k2, ← c93971] at e_1532_19
  have hr := e_1532_19
  simp only [e_1532_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14855)) := by
  have c93972 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93973 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93974 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93975 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93976 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93977 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_0 := arithEq_of_rows h (row := 1533) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_0
  simp only [← c93972, ← c93973, ← c93974] at e_1533_0
  have e_1533_1 := arithEq_of_rows h (row := 1533) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_1
  simp only [← c93975, ← c93976, k2, ← c93977] at e_1533_1
  have hr := e_1533_1
  simp only [e_1533_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14856)) := by
  have c93978 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93979 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93980 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93981 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93982 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93983 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_2 := arithEq_of_rows h (row := 1533) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_2
  simp only [← c93978, ← c93979, ← c93980] at e_1533_2
  have e_1533_3 := arithEq_of_rows h (row := 1533) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_3
  simp only [← c93981, ← c93982, k2, ← c93983] at e_1533_3
  have hr := e_1533_3
  simp only [e_1533_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14857)) := by
  have c93984 := (publicBatchWrapper89_copies2937 a h).1
  have c93985 := (publicBatchWrapper89_copies2937 a h).2.1
  have c93986 := (publicBatchWrapper89_copies2937 a h).2.2.1
  have c93987 := (publicBatchWrapper89_copies2937 a h).2.2.2.1
  have c93988 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.1
  have c93989 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_4 := arithEq_of_rows h (row := 1533) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_4
  simp only [← c93984, ← c93985, ← c93986] at e_1533_4
  have e_1533_5 := arithEq_of_rows h (row := 1533) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_5
  simp only [← c93987, ← c93988, k2, ← c93989] at e_1533_5
  have hr := e_1533_5
  simp only [e_1533_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14858)) := by
  have c93990 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.1
  have c93991 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.1
  have c93992 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.1
  have c93993 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.1
  have c93994 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93995 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_6 := arithEq_of_rows h (row := 1533) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_6
  simp only [← c93990, ← c93991, ← c93992] at e_1533_6
  have e_1533_7 := arithEq_of_rows h (row := 1533) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_7
  simp only [← c93993, ← c93994, k2, ← c93995] at e_1533_7
  have hr := e_1533_7
  simp only [e_1533_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14859)) := by
  have c93996 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93997 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93998 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93999 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94000 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94001 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_8 := arithEq_of_rows h (row := 1533) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_8
  simp only [← c93996, ← c93997, ← c93998] at e_1533_8
  have e_1533_9 := arithEq_of_rows h (row := 1533) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_9
  simp only [← c93999, ← c94000, k2, ← c94001] at e_1533_9
  have hr := e_1533_9
  simp only [e_1533_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14860)) := by
  have c94002 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94003 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94004 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94005 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94006 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94007 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_10 := arithEq_of_rows h (row := 1533) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_10
  simp only [← c94002, ← c94003, ← c94004] at e_1533_10
  have e_1533_11 := arithEq_of_rows h (row := 1533) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_11
  simp only [← c94005, ← c94006, k2, ← c94007] at e_1533_11
  have hr := e_1533_11
  simp only [e_1533_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14861)) := by
  have c94008 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94009 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94010 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94011 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94012 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94013 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_12 := arithEq_of_rows h (row := 1533) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_12
  simp only [← c94008, ← c94009, ← c94010] at e_1533_12
  have e_1533_13 := arithEq_of_rows h (row := 1533) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_13
  simp only [← c94011, ← c94012, k2, ← c94013] at e_1533_13
  have hr := e_1533_13
  simp only [e_1533_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14862)) := by
  have c94014 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94015 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94016 := (publicBatchWrapper89_copies2938 a h).1
  have c94017 := (publicBatchWrapper89_copies2938 a h).2.1
  have c94018 := (publicBatchWrapper89_copies2938 a h).2.2.1
  have c94019 := (publicBatchWrapper89_copies2938 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_14 := arithEq_of_rows h (row := 1533) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_14
  simp only [← c94014, ← c94015, ← c94016] at e_1533_14
  have e_1533_15 := arithEq_of_rows h (row := 1533) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_15
  simp only [← c94017, ← c94018, k2, ← c94019] at e_1533_15
  have hr := e_1533_15
  simp only [e_1533_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14863)) := by
  have c94020 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.1
  have c94021 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.1
  have c94022 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.1
  have c94023 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.1
  have c94024 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.1
  have c94025 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_16 := arithEq_of_rows h (row := 1533) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_16
  simp only [← c94020, ← c94021, ← c94022] at e_1533_16
  have e_1533_17 := arithEq_of_rows h (row := 1533) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_17
  simp only [← c94023, ← c94024, k2, ← c94025] at e_1533_17
  have hr := e_1533_17
  simp only [e_1533_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14864)) := by
  have c94026 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94027 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94028 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94029 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94030 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94031 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_18 := arithEq_of_rows h (row := 1533) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_18
  simp only [← c94026, ← c94027, ← c94028] at e_1533_18
  have e_1533_19 := arithEq_of_rows h (row := 1533) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_19
  simp only [← c94029, ← c94030, k2, ← c94031] at e_1533_19
  have hr := e_1533_19
  simp only [e_1533_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14865)) := by
  have c94032 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94033 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94034 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94035 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94036 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94037 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_0 := arithEq_of_rows h (row := 1534) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_0
  simp only [← c94032, ← c94033, ← c94034] at e_1534_0
  have e_1534_1 := arithEq_of_rows h (row := 1534) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_1
  simp only [← c94035, ← c94036, k2, ← c94037] at e_1534_1
  have hr := e_1534_1
  simp only [e_1534_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14866)) := by
  have c94038 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94039 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94040 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94041 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94042 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94043 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_2 := arithEq_of_rows h (row := 1534) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_2
  simp only [← c94038, ← c94039, ← c94040] at e_1534_2
  have e_1534_3 := arithEq_of_rows h (row := 1534) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_3
  simp only [← c94041, ← c94042, k2, ← c94043] at e_1534_3
  have hr := e_1534_3
  simp only [e_1534_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14867)) := by
  have c94044 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94045 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94046 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94047 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94048 := (publicBatchWrapper89_copies2939 a h).1
  have c94049 := (publicBatchWrapper89_copies2939 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_4 := arithEq_of_rows h (row := 1534) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_4
  simp only [← c94044, ← c94045, ← c94046] at e_1534_4
  have e_1534_5 := arithEq_of_rows h (row := 1534) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_5
  simp only [← c94047, ← c94048, k2, ← c94049] at e_1534_5
  have hr := e_1534_5
  simp only [e_1534_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14868)) := by
  have c94050 := (publicBatchWrapper89_copies2939 a h).2.2.1
  have c94051 := (publicBatchWrapper89_copies2939 a h).2.2.2.1
  have c94052 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.1
  have c94053 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.1
  have c94054 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.1
  have c94055 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_6 := arithEq_of_rows h (row := 1534) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_6
  simp only [← c94050, ← c94051, ← c94052] at e_1534_6
  have e_1534_7 := arithEq_of_rows h (row := 1534) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_7
  simp only [← c94053, ← c94054, k2, ← c94055] at e_1534_7
  have hr := e_1534_7
  simp only [e_1534_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14869)) := by
  have c94056 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.1
  have c94057 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.1
  have c94058 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94059 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94060 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94061 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_8 := arithEq_of_rows h (row := 1534) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_8
  simp only [← c94056, ← c94057, ← c94058] at e_1534_8
  have e_1534_9 := arithEq_of_rows h (row := 1534) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_9
  simp only [← c94059, ← c94060, k2, ← c94061] at e_1534_9
  have hr := e_1534_9
  simp only [e_1534_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14870)) := by
  have c94062 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94063 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94064 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94065 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94066 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94067 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_10 := arithEq_of_rows h (row := 1534) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_10
  simp only [← c94062, ← c94063, ← c94064] at e_1534_10
  have e_1534_11 := arithEq_of_rows h (row := 1534) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_11
  simp only [← c94065, ← c94066, k2, ← c94067] at e_1534_11
  have hr := e_1534_11
  simp only [e_1534_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14871)) := by
  have c94068 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94069 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94070 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94071 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94072 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94073 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_12 := arithEq_of_rows h (row := 1534) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_12
  simp only [← c94068, ← c94069, ← c94070] at e_1534_12
  have e_1534_13 := arithEq_of_rows h (row := 1534) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_13
  simp only [← c94071, ← c94072, k2, ← c94073] at e_1534_13
  have hr := e_1534_13
  simp only [e_1534_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14872)) := by
  have c94074 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94075 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94076 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94077 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94078 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94079 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_14 := arithEq_of_rows h (row := 1534) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_14
  simp only [← c94074, ← c94075, ← c94076] at e_1534_14
  have e_1534_15 := arithEq_of_rows h (row := 1534) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_15
  simp only [← c94077, ← c94078, k2, ← c94079] at e_1534_15
  have hr := e_1534_15
  simp only [e_1534_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14873)) := by
  have c94080 := (publicBatchWrapper89_copies2940 a h).1
  have c94081 := (publicBatchWrapper89_copies2940 a h).2.1
  have c94082 := (publicBatchWrapper89_copies2940 a h).2.2.1
  have c94083 := (publicBatchWrapper89_copies2940 a h).2.2.2.1
  have c94084 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.1
  have c94085 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_16 := arithEq_of_rows h (row := 1534) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_16
  simp only [← c94080, ← c94081, ← c94082] at e_1534_16
  have e_1534_17 := arithEq_of_rows h (row := 1534) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_17
  simp only [← c94083, ← c94084, k2, ← c94085] at e_1534_17
  have hr := e_1534_17
  simp only [e_1534_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14874)) := by
  have c94086 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.1
  have c94087 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.1
  have c94088 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.1
  have c94089 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.1
  have c94090 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94091 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_18 := arithEq_of_rows h (row := 1534) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_18
  simp only [← c94086, ← c94087, ← c94088] at e_1534_18
  have e_1534_19 := arithEq_of_rows h (row := 1534) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_19
  simp only [← c94089, ← c94090, k2, ← c94091] at e_1534_19
  have hr := e_1534_19
  simp only [e_1534_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14875)) := by
  have c94092 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94093 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94094 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94095 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94096 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94097 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_0 := arithEq_of_rows h (row := 1535) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_0
  simp only [← c94092, ← c94093, ← c94094] at e_1535_0
  have e_1535_1 := arithEq_of_rows h (row := 1535) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_1
  simp only [← c94095, ← c94096, k2, ← c94097] at e_1535_1
  have hr := e_1535_1
  simp only [e_1535_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14876)) := by
  have c94098 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94099 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94100 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94101 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94102 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94103 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_2 := arithEq_of_rows h (row := 1535) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_2
  simp only [← c94098, ← c94099, ← c94100] at e_1535_2
  have e_1535_3 := arithEq_of_rows h (row := 1535) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_3
  simp only [← c94101, ← c94102, k2, ← c94103] at e_1535_3
  have hr := e_1535_3
  simp only [e_1535_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14877)) := by
  have c94104 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94105 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94106 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94107 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94108 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94109 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_4 := arithEq_of_rows h (row := 1535) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_4
  simp only [← c94104, ← c94105, ← c94106] at e_1535_4
  have e_1535_5 := arithEq_of_rows h (row := 1535) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_5
  simp only [← c94107, ← c94108, k2, ← c94109] at e_1535_5
  have hr := e_1535_5
  simp only [e_1535_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14878)) := by
  have c94110 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94111 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94112 := (publicBatchWrapper89_copies2941 a h).1
  have c94113 := (publicBatchWrapper89_copies2941 a h).2.1
  have c94114 := (publicBatchWrapper89_copies2941 a h).2.2.1
  have c94115 := (publicBatchWrapper89_copies2941 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_6 := arithEq_of_rows h (row := 1535) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_6
  simp only [← c94110, ← c94111, ← c94112] at e_1535_6
  have e_1535_7 := arithEq_of_rows h (row := 1535) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_7
  simp only [← c94113, ← c94114, k2, ← c94115] at e_1535_7
  have hr := e_1535_7
  simp only [e_1535_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14879)) := by
  have c94116 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.1
  have c94117 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.1
  have c94118 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.1
  have c94119 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.1
  have c94120 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.1
  have c94121 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_8 := arithEq_of_rows h (row := 1535) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_8
  simp only [← c94116, ← c94117, ← c94118] at e_1535_8
  have e_1535_9 := arithEq_of_rows h (row := 1535) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_9
  simp only [← c94119, ← c94120, k2, ← c94121] at e_1535_9
  have hr := e_1535_9
  simp only [e_1535_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14880)) := by
  have c94122 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94123 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94124 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94125 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94126 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94127 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_10 := arithEq_of_rows h (row := 1535) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_10
  simp only [← c94122, ← c94123, ← c94124] at e_1535_10
  have e_1535_11 := arithEq_of_rows h (row := 1535) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_11
  simp only [← c94125, ← c94126, k2, ← c94127] at e_1535_11
  have hr := e_1535_11
  simp only [e_1535_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14881)) := by
  have c94128 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94129 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94130 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94131 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94132 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94133 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_12 := arithEq_of_rows h (row := 1535) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_12
  simp only [← c94128, ← c94129, ← c94130] at e_1535_12
  have e_1535_13 := arithEq_of_rows h (row := 1535) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_13
  simp only [← c94131, ← c94132, k2, ← c94133] at e_1535_13
  have hr := e_1535_13
  simp only [e_1535_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14882)) := by
  have c94134 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94135 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94136 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94137 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94138 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94139 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_14 := arithEq_of_rows h (row := 1535) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_14
  simp only [← c94134, ← c94135, ← c94136] at e_1535_14
  have e_1535_15 := arithEq_of_rows h (row := 1535) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_15
  simp only [← c94137, ← c94138, k2, ← c94139] at e_1535_15
  have hr := e_1535_15
  simp only [e_1535_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14883)) := by
  have c94140 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94141 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94142 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94143 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94144 := (publicBatchWrapper89_copies2942 a h).1
  have c94145 := (publicBatchWrapper89_copies2942 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_16 := arithEq_of_rows h (row := 1535) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_16
  simp only [← c94140, ← c94141, ← c94142] at e_1535_16
  have e_1535_17 := arithEq_of_rows h (row := 1535) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_17
  simp only [← c94143, ← c94144, k2, ← c94145] at e_1535_17
  have hr := e_1535_17
  simp only [e_1535_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14884)) := by
  have c94146 := (publicBatchWrapper89_copies2942 a h).2.2.1
  have c94147 := (publicBatchWrapper89_copies2942 a h).2.2.2.1
  have c94148 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.1
  have c94149 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.1
  have c94150 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.1
  have c94151 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_18 := arithEq_of_rows h (row := 1535) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_18
  simp only [← c94146, ← c94147, ← c94148] at e_1535_18
  have e_1535_19 := arithEq_of_rows h (row := 1535) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_19
  simp only [← c94149, ← c94150, k2, ← c94151] at e_1535_19
  have hr := e_1535_19
  simp only [e_1535_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14885)) := by
  have c94152 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.1
  have c94153 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.1
  have c94154 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94155 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94156 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94157 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_0 := arithEq_of_rows h (row := 1536) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_0
  simp only [← c94152, ← c94153, ← c94154] at e_1536_0
  have e_1536_1 := arithEq_of_rows h (row := 1536) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_1
  simp only [← c94155, ← c94156, k2, ← c94157] at e_1536_1
  have hr := e_1536_1
  simp only [e_1536_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14886)) := by
  have c94158 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94159 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94160 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94161 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94162 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94163 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_2 := arithEq_of_rows h (row := 1536) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_2
  simp only [← c94158, ← c94159, ← c94160] at e_1536_2
  have e_1536_3 := arithEq_of_rows h (row := 1536) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_3
  simp only [← c94161, ← c94162, k2, ← c94163] at e_1536_3
  have hr := e_1536_3
  simp only [e_1536_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14887)) := by
  have c94164 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94165 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94166 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94167 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94168 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94169 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_4 := arithEq_of_rows h (row := 1536) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_4
  simp only [← c94164, ← c94165, ← c94166] at e_1536_4
  have e_1536_5 := arithEq_of_rows h (row := 1536) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_5
  simp only [← c94167, ← c94168, k2, ← c94169] at e_1536_5
  have hr := e_1536_5
  simp only [e_1536_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14888)) := by
  have c94170 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94171 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94172 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94173 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94174 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94175 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_6 := arithEq_of_rows h (row := 1536) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_6
  simp only [← c94170, ← c94171, ← c94172] at e_1536_6
  have e_1536_7 := arithEq_of_rows h (row := 1536) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_7
  simp only [← c94173, ← c94174, k2, ← c94175] at e_1536_7
  have hr := e_1536_7
  simp only [e_1536_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14889)) := by
  have c94176 := (publicBatchWrapper89_copies2943 a h).1
  have c94177 := (publicBatchWrapper89_copies2943 a h).2.1
  have c94178 := (publicBatchWrapper89_copies2943 a h).2.2.1
  have c94179 := (publicBatchWrapper89_copies2943 a h).2.2.2.1
  have c94180 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.1
  have c94181 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_8 := arithEq_of_rows h (row := 1536) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_8
  simp only [← c94176, ← c94177, ← c94178] at e_1536_8
  have e_1536_9 := arithEq_of_rows h (row := 1536) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_9
  simp only [← c94179, ← c94180, k2, ← c94181] at e_1536_9
  have hr := e_1536_9
  simp only [e_1536_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14890)) := by
  have c94182 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.1
  have c94183 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.1
  have c94184 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.1
  have c94185 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.1
  have c94186 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94187 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_10 := arithEq_of_rows h (row := 1536) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_10
  simp only [← c94182, ← c94183, ← c94184] at e_1536_10
  have e_1536_11 := arithEq_of_rows h (row := 1536) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_11
  simp only [← c94185, ← c94186, k2, ← c94187] at e_1536_11
  have hr := e_1536_11
  simp only [e_1536_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14891)) := by
  have c94188 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94189 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94190 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94191 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94192 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94193 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_12 := arithEq_of_rows h (row := 1536) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_12
  simp only [← c94188, ← c94189, ← c94190] at e_1536_12
  have e_1536_13 := arithEq_of_rows h (row := 1536) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_13
  simp only [← c94191, ← c94192, k2, ← c94193] at e_1536_13
  have hr := e_1536_13
  simp only [e_1536_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14892)) := by
  have c94194 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94195 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94196 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94197 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94198 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94199 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_14 := arithEq_of_rows h (row := 1536) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_14
  simp only [← c94194, ← c94195, ← c94196] at e_1536_14
  have e_1536_15 := arithEq_of_rows h (row := 1536) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_15
  simp only [← c94197, ← c94198, k2, ← c94199] at e_1536_15
  have hr := e_1536_15
  simp only [e_1536_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14893)) := by
  have c94200 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94201 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94202 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94203 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94204 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94205 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_16 := arithEq_of_rows h (row := 1536) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_16
  simp only [← c94200, ← c94201, ← c94202] at e_1536_16
  have e_1536_17 := arithEq_of_rows h (row := 1536) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_17
  simp only [← c94203, ← c94204, k2, ← c94205] at e_1536_17
  have hr := e_1536_17
  simp only [e_1536_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14894)) := by
  have c94206 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94207 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94208 := (publicBatchWrapper89_copies2944 a h).1
  have c94209 := (publicBatchWrapper89_copies2944 a h).2.1
  have c94210 := (publicBatchWrapper89_copies2944 a h).2.2.1
  have c94211 := (publicBatchWrapper89_copies2944 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_18 := arithEq_of_rows h (row := 1536) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_18
  simp only [← c94206, ← c94207, ← c94208] at e_1536_18
  have e_1536_19 := arithEq_of_rows h (row := 1536) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_19
  simp only [← c94209, ← c94210, k2, ← c94211] at e_1536_19
  have hr := e_1536_19
  simp only [e_1536_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14895)) := by
  have c94212 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.1
  have c94213 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.1
  have c94214 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.1
  have c94215 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.1
  have c94216 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.1
  have c94217 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_0 := arithEq_of_rows h (row := 1537) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_0
  simp only [← c94212, ← c94213, ← c94214] at e_1537_0
  have e_1537_1 := arithEq_of_rows h (row := 1537) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_1
  simp only [← c94215, ← c94216, k2, ← c94217] at e_1537_1
  have hr := e_1537_1
  simp only [e_1537_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14896)) := by
  have c94218 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94219 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94220 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94221 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94222 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94223 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_2 := arithEq_of_rows h (row := 1537) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_2
  simp only [← c94218, ← c94219, ← c94220] at e_1537_2
  have e_1537_3 := arithEq_of_rows h (row := 1537) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_3
  simp only [← c94221, ← c94222, k2, ← c94223] at e_1537_3
  have hr := e_1537_3
  simp only [e_1537_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14897)) := by
  have c94224 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94225 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94226 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94227 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94228 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94229 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_4 := arithEq_of_rows h (row := 1537) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_4
  simp only [← c94224, ← c94225, ← c94226] at e_1537_4
  have e_1537_5 := arithEq_of_rows h (row := 1537) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_5
  simp only [← c94227, ← c94228, k2, ← c94229] at e_1537_5
  have hr := e_1537_5
  simp only [e_1537_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14898)) := by
  have c94230 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94231 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94232 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94233 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94234 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94235 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_6 := arithEq_of_rows h (row := 1537) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_6
  simp only [← c94230, ← c94231, ← c94232] at e_1537_6
  have e_1537_7 := arithEq_of_rows h (row := 1537) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_7
  simp only [← c94233, ← c94234, k2, ← c94235] at e_1537_7
  have hr := e_1537_7
  simp only [e_1537_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14899)) := by
  have c94236 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94237 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94238 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94239 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94240 := (publicBatchWrapper89_copies2945 a h).1
  have c94241 := (publicBatchWrapper89_copies2945 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_8 := arithEq_of_rows h (row := 1537) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_8
  simp only [← c94236, ← c94237, ← c94238] at e_1537_8
  have e_1537_9 := arithEq_of_rows h (row := 1537) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_9
  simp only [← c94239, ← c94240, k2, ← c94241] at e_1537_9
  have hr := e_1537_9
  simp only [e_1537_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14900)) := by
  have c94242 := (publicBatchWrapper89_copies2945 a h).2.2.1
  have c94243 := (publicBatchWrapper89_copies2945 a h).2.2.2.1
  have c94244 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.1
  have c94245 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.1
  have c94246 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.1
  have c94247 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_10 := arithEq_of_rows h (row := 1537) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_10
  simp only [← c94242, ← c94243, ← c94244] at e_1537_10
  have e_1537_11 := arithEq_of_rows h (row := 1537) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_11
  simp only [← c94245, ← c94246, k2, ← c94247] at e_1537_11
  have hr := e_1537_11
  simp only [e_1537_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14901)) := by
  have c94248 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.1
  have c94249 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.1
  have c94250 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94251 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94252 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94253 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_12 := arithEq_of_rows h (row := 1537) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_12
  simp only [← c94248, ← c94249, ← c94250] at e_1537_12
  have e_1537_13 := arithEq_of_rows h (row := 1537) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_13
  simp only [← c94251, ← c94252, k2, ← c94253] at e_1537_13
  have hr := e_1537_13
  simp only [e_1537_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14902)) := by
  have c94254 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94255 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94256 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94257 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94258 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94259 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_14 := arithEq_of_rows h (row := 1537) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_14
  simp only [← c94254, ← c94255, ← c94256] at e_1537_14
  have e_1537_15 := arithEq_of_rows h (row := 1537) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_15
  simp only [← c94257, ← c94258, k2, ← c94259] at e_1537_15
  have hr := e_1537_15
  simp only [e_1537_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14903)) := by
  have c94260 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94261 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94262 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94263 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94264 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94265 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_16 := arithEq_of_rows h (row := 1537) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_16
  simp only [← c94260, ← c94261, ← c94262] at e_1537_16
  have e_1537_17 := arithEq_of_rows h (row := 1537) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_17
  simp only [← c94263, ← c94264, k2, ← c94265] at e_1537_17
  have hr := e_1537_17
  simp only [e_1537_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14904)) := by
  have c94266 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94267 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94268 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94269 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94270 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94271 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_18 := arithEq_of_rows h (row := 1537) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_18
  simp only [← c94266, ← c94267, ← c94268] at e_1537_18
  have e_1537_19 := arithEq_of_rows h (row := 1537) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_19
  simp only [← c94269, ← c94270, k2, ← c94271] at e_1537_19
  have hr := e_1537_19
  simp only [e_1537_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14905)) := by
  have c94272 := (publicBatchWrapper89_copies2946 a h).1
  have c94273 := (publicBatchWrapper89_copies2946 a h).2.1
  have c94274 := (publicBatchWrapper89_copies2946 a h).2.2.1
  have c94275 := (publicBatchWrapper89_copies2946 a h).2.2.2.1
  have c94276 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.1
  have c94277 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_0 := arithEq_of_rows h (row := 1538) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_0
  simp only [← c94272, ← c94273, ← c94274] at e_1538_0
  have e_1538_1 := arithEq_of_rows h (row := 1538) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_1
  simp only [← c94275, ← c94276, k2, ← c94277] at e_1538_1
  have hr := e_1538_1
  simp only [e_1538_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14906)) := by
  have c94278 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.1
  have c94279 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.1
  have c94280 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.1
  have c94281 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.1
  have c94282 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94283 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_2 := arithEq_of_rows h (row := 1538) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_2
  simp only [← c94278, ← c94279, ← c94280] at e_1538_2
  have e_1538_3 := arithEq_of_rows h (row := 1538) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_3
  simp only [← c94281, ← c94282, k2, ← c94283] at e_1538_3
  have hr := e_1538_3
  simp only [e_1538_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14907)) := by
  have c94284 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94285 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94286 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94287 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94288 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94289 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_4 := arithEq_of_rows h (row := 1538) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_4
  simp only [← c94284, ← c94285, ← c94286] at e_1538_4
  have e_1538_5 := arithEq_of_rows h (row := 1538) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_5
  simp only [← c94287, ← c94288, k2, ← c94289] at e_1538_5
  have hr := e_1538_5
  simp only [e_1538_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14908)) := by
  have c94290 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94291 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94292 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94293 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94294 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94295 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_6 := arithEq_of_rows h (row := 1538) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_6
  simp only [← c94290, ← c94291, ← c94292] at e_1538_6
  have e_1538_7 := arithEq_of_rows h (row := 1538) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_7
  simp only [← c94293, ← c94294, k2, ← c94295] at e_1538_7
  have hr := e_1538_7
  simp only [e_1538_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14909)) := by
  have c94296 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94297 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94298 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94299 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94300 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94301 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_8 := arithEq_of_rows h (row := 1538) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_8
  simp only [← c94296, ← c94297, ← c94298] at e_1538_8
  have e_1538_9 := arithEq_of_rows h (row := 1538) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_9
  simp only [← c94299, ← c94300, k2, ← c94301] at e_1538_9
  have hr := e_1538_9
  simp only [e_1538_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14910)) := by
  have c94302 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94303 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94304 := (publicBatchWrapper89_copies2947 a h).1
  have c94305 := (publicBatchWrapper89_copies2947 a h).2.1
  have c94306 := (publicBatchWrapper89_copies2947 a h).2.2.1
  have c94307 := (publicBatchWrapper89_copies2947 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_10 := arithEq_of_rows h (row := 1538) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_10
  simp only [← c94302, ← c94303, ← c94304] at e_1538_10
  have e_1538_11 := arithEq_of_rows h (row := 1538) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_11
  simp only [← c94305, ← c94306, k2, ← c94307] at e_1538_11
  have hr := e_1538_11
  simp only [e_1538_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14911)) := by
  have c94308 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.1
  have c94309 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.1
  have c94310 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.1
  have c94311 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.1
  have c94312 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.1
  have c94313 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_12 := arithEq_of_rows h (row := 1538) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_12
  simp only [← c94308, ← c94309, ← c94310] at e_1538_12
  have e_1538_13 := arithEq_of_rows h (row := 1538) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_13
  simp only [← c94311, ← c94312, k2, ← c94313] at e_1538_13
  have hr := e_1538_13
  simp only [e_1538_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14912)) := by
  have c94314 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94315 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94316 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94317 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94318 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94319 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_14 := arithEq_of_rows h (row := 1538) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_14
  simp only [← c94314, ← c94315, ← c94316] at e_1538_14
  have e_1538_15 := arithEq_of_rows h (row := 1538) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_15
  simp only [← c94317, ← c94318, k2, ← c94319] at e_1538_15
  have hr := e_1538_15
  simp only [e_1538_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14913)) := by
  have c94320 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94321 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94322 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94323 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94324 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94325 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_16 := arithEq_of_rows h (row := 1538) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_16
  simp only [← c94320, ← c94321, ← c94322] at e_1538_16
  have e_1538_17 := arithEq_of_rows h (row := 1538) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_17
  simp only [← c94323, ← c94324, k2, ← c94325] at e_1538_17
  have hr := e_1538_17
  simp only [e_1538_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14914)) := by
  have c94326 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94327 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94328 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94329 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94330 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94331 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_18 := arithEq_of_rows h (row := 1538) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_18
  simp only [← c94326, ← c94327, ← c94328] at e_1538_18
  have e_1538_19 := arithEq_of_rows h (row := 1538) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_19
  simp only [← c94329, ← c94330, k2, ← c94331] at e_1538_19
  have hr := e_1538_19
  simp only [e_1538_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14915)) := by
  have c94332 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94333 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94334 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94335 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94336 := (publicBatchWrapper89_copies2948 a h).1
  have c94337 := (publicBatchWrapper89_copies2948 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_0 := arithEq_of_rows h (row := 1539) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_0
  simp only [← c94332, ← c94333, ← c94334] at e_1539_0
  have e_1539_1 := arithEq_of_rows h (row := 1539) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_1
  simp only [← c94335, ← c94336, k2, ← c94337] at e_1539_1
  have hr := e_1539_1
  simp only [e_1539_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14916)) := by
  have c94338 := (publicBatchWrapper89_copies2948 a h).2.2.1
  have c94339 := (publicBatchWrapper89_copies2948 a h).2.2.2.1
  have c94340 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.1
  have c94341 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.1
  have c94342 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.1
  have c94343 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_2 := arithEq_of_rows h (row := 1539) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_2
  simp only [← c94338, ← c94339, ← c94340] at e_1539_2
  have e_1539_3 := arithEq_of_rows h (row := 1539) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_3
  simp only [← c94341, ← c94342, k2, ← c94343] at e_1539_3
  have hr := e_1539_3
  simp only [e_1539_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14917)) := by
  have c94344 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.1
  have c94345 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.1
  have c94346 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94347 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94348 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94349 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_4 := arithEq_of_rows h (row := 1539) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_4
  simp only [← c94344, ← c94345, ← c94346] at e_1539_4
  have e_1539_5 := arithEq_of_rows h (row := 1539) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_5
  simp only [← c94347, ← c94348, k2, ← c94349] at e_1539_5
  have hr := e_1539_5
  simp only [e_1539_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14918)) := by
  have c94350 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94351 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94352 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94353 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94354 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94355 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_6 := arithEq_of_rows h (row := 1539) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_6
  simp only [← c94350, ← c94351, ← c94352] at e_1539_6
  have e_1539_7 := arithEq_of_rows h (row := 1539) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_7
  simp only [← c94353, ← c94354, k2, ← c94355] at e_1539_7
  have hr := e_1539_7
  simp only [e_1539_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14919)) := by
  have c94356 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94357 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94358 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94359 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94360 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94361 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_8 := arithEq_of_rows h (row := 1539) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_8
  simp only [← c94356, ← c94357, ← c94358] at e_1539_8
  have e_1539_9 := arithEq_of_rows h (row := 1539) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_9
  simp only [← c94359, ← c94360, k2, ← c94361] at e_1539_9
  have hr := e_1539_9
  simp only [e_1539_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14920)) := by
  have c94362 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94363 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94364 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94365 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94366 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94367 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_10 := arithEq_of_rows h (row := 1539) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_10
  simp only [← c94362, ← c94363, ← c94364] at e_1539_10
  have e_1539_11 := arithEq_of_rows h (row := 1539) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_11
  simp only [← c94365, ← c94366, k2, ← c94367] at e_1539_11
  have hr := e_1539_11
  simp only [e_1539_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14921)) := by
  have c94368 := (publicBatchWrapper89_copies2949 a h).1
  have c94369 := (publicBatchWrapper89_copies2949 a h).2.1
  have c94370 := (publicBatchWrapper89_copies2949 a h).2.2.1
  have c94371 := (publicBatchWrapper89_copies2949 a h).2.2.2.1
  have c94372 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.1
  have c94373 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_12 := arithEq_of_rows h (row := 1539) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_12
  simp only [← c94368, ← c94369, ← c94370] at e_1539_12
  have e_1539_13 := arithEq_of_rows h (row := 1539) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_13
  simp only [← c94371, ← c94372, k2, ← c94373] at e_1539_13
  have hr := e_1539_13
  simp only [e_1539_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14922)) := by
  have c94374 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.1
  have c94375 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.1
  have c94376 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.1
  have c94377 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.1
  have c94378 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94379 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_14 := arithEq_of_rows h (row := 1539) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_14
  simp only [← c94374, ← c94375, ← c94376] at e_1539_14
  have e_1539_15 := arithEq_of_rows h (row := 1539) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_15
  simp only [← c94377, ← c94378, k2, ← c94379] at e_1539_15
  have hr := e_1539_15
  simp only [e_1539_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14923)) := by
  have c94380 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94381 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94382 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94383 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94384 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94385 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_16 := arithEq_of_rows h (row := 1539) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_16
  simp only [← c94380, ← c94381, ← c94382] at e_1539_16
  have e_1539_17 := arithEq_of_rows h (row := 1539) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_17
  simp only [← c94383, ← c94384, k2, ← c94385] at e_1539_17
  have hr := e_1539_17
  simp only [e_1539_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14924)) := by
  have c94386 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94387 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94388 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94389 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94390 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94391 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_18 := arithEq_of_rows h (row := 1539) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_18
  simp only [← c94386, ← c94387, ← c94388] at e_1539_18
  have e_1539_19 := arithEq_of_rows h (row := 1539) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_19
  simp only [← c94389, ← c94390, k2, ← c94391] at e_1539_19
  have hr := e_1539_19
  simp only [e_1539_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14925)) := by
  have c94392 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94393 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94394 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94395 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94396 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94397 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_0 := arithEq_of_rows h (row := 1540) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_0
  simp only [← c94392, ← c94393, ← c94394] at e_1540_0
  have e_1540_1 := arithEq_of_rows h (row := 1540) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_1
  simp only [← c94395, ← c94396, k2, ← c94397] at e_1540_1
  have hr := e_1540_1
  simp only [e_1540_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14926)) := by
  have c94398 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94399 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94400 := (publicBatchWrapper89_copies2950 a h).1
  have c94401 := (publicBatchWrapper89_copies2950 a h).2.1
  have c94402 := (publicBatchWrapper89_copies2950 a h).2.2.1
  have c94403 := (publicBatchWrapper89_copies2950 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_2 := arithEq_of_rows h (row := 1540) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_2
  simp only [← c94398, ← c94399, ← c94400] at e_1540_2
  have e_1540_3 := arithEq_of_rows h (row := 1540) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_3
  simp only [← c94401, ← c94402, k2, ← c94403] at e_1540_3
  have hr := e_1540_3
  simp only [e_1540_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14927)) := by
  have c94404 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.1
  have c94405 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.1
  have c94406 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.1
  have c94407 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.1
  have c94408 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.1
  have c94409 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_4 := arithEq_of_rows h (row := 1540) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_4
  simp only [← c94404, ← c94405, ← c94406] at e_1540_4
  have e_1540_5 := arithEq_of_rows h (row := 1540) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_5
  simp only [← c94407, ← c94408, k2, ← c94409] at e_1540_5
  have hr := e_1540_5
  simp only [e_1540_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14928)) := by
  have c94410 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94411 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94412 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94413 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94414 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94415 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_6 := arithEq_of_rows h (row := 1540) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_6
  simp only [← c94410, ← c94411, ← c94412] at e_1540_6
  have e_1540_7 := arithEq_of_rows h (row := 1540) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_7
  simp only [← c94413, ← c94414, k2, ← c94415] at e_1540_7
  have hr := e_1540_7
  simp only [e_1540_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14929)) := by
  have c94416 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94417 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94418 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94419 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94420 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94421 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_8 := arithEq_of_rows h (row := 1540) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_8
  simp only [← c94416, ← c94417, ← c94418] at e_1540_8
  have e_1540_9 := arithEq_of_rows h (row := 1540) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_9
  simp only [← c94419, ← c94420, k2, ← c94421] at e_1540_9
  have hr := e_1540_9
  simp only [e_1540_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14930)) := by
  have c94422 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94423 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94424 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94425 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94426 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94427 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_10 := arithEq_of_rows h (row := 1540) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_10
  simp only [← c94422, ← c94423, ← c94424] at e_1540_10
  have e_1540_11 := arithEq_of_rows h (row := 1540) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_11
  simp only [← c94425, ← c94426, k2, ← c94427] at e_1540_11
  have hr := e_1540_11
  simp only [e_1540_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14931)) := by
  have c94428 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94429 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94430 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94431 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94432 := (publicBatchWrapper89_copies2951 a h).1
  have c94433 := (publicBatchWrapper89_copies2951 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_12 := arithEq_of_rows h (row := 1540) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_12
  simp only [← c94428, ← c94429, ← c94430] at e_1540_12
  have e_1540_13 := arithEq_of_rows h (row := 1540) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_13
  simp only [← c94431, ← c94432, k2, ← c94433] at e_1540_13
  have hr := e_1540_13
  simp only [e_1540_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14932)) := by
  have c94434 := (publicBatchWrapper89_copies2951 a h).2.2.1
  have c94435 := (publicBatchWrapper89_copies2951 a h).2.2.2.1
  have c94436 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.1
  have c94437 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.1
  have c94438 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.1
  have c94439 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_14 := arithEq_of_rows h (row := 1540) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_14
  simp only [← c94434, ← c94435, ← c94436] at e_1540_14
  have e_1540_15 := arithEq_of_rows h (row := 1540) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_15
  simp only [← c94437, ← c94438, k2, ← c94439] at e_1540_15
  have hr := e_1540_15
  simp only [e_1540_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14933)) := by
  have c94440 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.1
  have c94441 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.1
  have c94442 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94443 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94444 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94445 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_16 := arithEq_of_rows h (row := 1540) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_16
  simp only [← c94440, ← c94441, ← c94442] at e_1540_16
  have e_1540_17 := arithEq_of_rows h (row := 1540) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_17
  simp only [← c94443, ← c94444, k2, ← c94445] at e_1540_17
  have hr := e_1540_17
  simp only [e_1540_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14934)) := by
  have c94446 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94447 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94448 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94449 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94450 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94451 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_18 := arithEq_of_rows h (row := 1540) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_18
  simp only [← c94446, ← c94447, ← c94448] at e_1540_18
  have e_1540_19 := arithEq_of_rows h (row := 1540) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_19
  simp only [← c94449, ← c94450, k2, ← c94451] at e_1540_19
  have hr := e_1540_19
  simp only [e_1540_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14935)) := by
  have c94452 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94453 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94454 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94455 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94456 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94457 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_0 := arithEq_of_rows h (row := 1541) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_0
  simp only [← c94452, ← c94453, ← c94454] at e_1541_0
  have e_1541_1 := arithEq_of_rows h (row := 1541) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_1
  simp only [← c94455, ← c94456, k2, ← c94457] at e_1541_1
  have hr := e_1541_1
  simp only [e_1541_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14936)) := by
  have c94458 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94459 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94460 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94461 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94462 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94463 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_2 := arithEq_of_rows h (row := 1541) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_2
  simp only [← c94458, ← c94459, ← c94460] at e_1541_2
  have e_1541_3 := arithEq_of_rows h (row := 1541) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_3
  simp only [← c94461, ← c94462, k2, ← c94463] at e_1541_3
  have hr := e_1541_3
  simp only [e_1541_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14937)) := by
  have c94464 := (publicBatchWrapper89_copies2952 a h).1
  have c94465 := (publicBatchWrapper89_copies2952 a h).2.1
  have c94466 := (publicBatchWrapper89_copies2952 a h).2.2.1
  have c94467 := (publicBatchWrapper89_copies2952 a h).2.2.2.1
  have c94468 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.1
  have c94469 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_4 := arithEq_of_rows h (row := 1541) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_4
  simp only [← c94464, ← c94465, ← c94466] at e_1541_4
  have e_1541_5 := arithEq_of_rows h (row := 1541) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_5
  simp only [← c94467, ← c94468, k2, ← c94469] at e_1541_5
  have hr := e_1541_5
  simp only [e_1541_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14938)) := by
  have c94470 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.1
  have c94471 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.1
  have c94472 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.1
  have c94473 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.1
  have c94474 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94475 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_6 := arithEq_of_rows h (row := 1541) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_6
  simp only [← c94470, ← c94471, ← c94472] at e_1541_6
  have e_1541_7 := arithEq_of_rows h (row := 1541) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_7
  simp only [← c94473, ← c94474, k2, ← c94475] at e_1541_7
  have hr := e_1541_7
  simp only [e_1541_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14939)) := by
  have c94476 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94477 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94478 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94479 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94480 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94481 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_8 := arithEq_of_rows h (row := 1541) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_8
  simp only [← c94476, ← c94477, ← c94478] at e_1541_8
  have e_1541_9 := arithEq_of_rows h (row := 1541) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_9
  simp only [← c94479, ← c94480, k2, ← c94481] at e_1541_9
  have hr := e_1541_9
  simp only [e_1541_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14940)) := by
  have c94482 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94483 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94484 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94485 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94486 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94487 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_10 := arithEq_of_rows h (row := 1541) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_10
  simp only [← c94482, ← c94483, ← c94484] at e_1541_10
  have e_1541_11 := arithEq_of_rows h (row := 1541) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_11
  simp only [← c94485, ← c94486, k2, ← c94487] at e_1541_11
  have hr := e_1541_11
  simp only [e_1541_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14941)) := by
  have c94488 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94489 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94490 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94491 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94492 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94493 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_12 := arithEq_of_rows h (row := 1541) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_12
  simp only [← c94488, ← c94489, ← c94490] at e_1541_12
  have e_1541_13 := arithEq_of_rows h (row := 1541) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_13
  simp only [← c94491, ← c94492, k2, ← c94493] at e_1541_13
  have hr := e_1541_13
  simp only [e_1541_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14942)) := by
  have c94494 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94495 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94496 := (publicBatchWrapper89_copies2953 a h).1
  have c94497 := (publicBatchWrapper89_copies2953 a h).2.1
  have c94498 := (publicBatchWrapper89_copies2953 a h).2.2.1
  have c94499 := (publicBatchWrapper89_copies2953 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_14 := arithEq_of_rows h (row := 1541) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_14
  simp only [← c94494, ← c94495, ← c94496] at e_1541_14
  have e_1541_15 := arithEq_of_rows h (row := 1541) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_15
  simp only [← c94497, ← c94498, k2, ← c94499] at e_1541_15
  have hr := e_1541_15
  simp only [e_1541_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14943)) := by
  have c94500 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.1
  have c94501 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.1
  have c94502 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.1
  have c94503 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.1
  have c94504 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.1
  have c94505 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_16 := arithEq_of_rows h (row := 1541) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_16
  simp only [← c94500, ← c94501, ← c94502] at e_1541_16
  have e_1541_17 := arithEq_of_rows h (row := 1541) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_17
  simp only [← c94503, ← c94504, k2, ← c94505] at e_1541_17
  have hr := e_1541_17
  simp only [e_1541_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14944)) := by
  have c94506 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94507 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94508 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94509 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94510 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94511 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_18 := arithEq_of_rows h (row := 1541) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_18
  simp only [← c94506, ← c94507, ← c94508] at e_1541_18
  have e_1541_19 := arithEq_of_rows h (row := 1541) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_19
  simp only [← c94509, ← c94510, k2, ← c94511] at e_1541_19
  have hr := e_1541_19
  simp only [e_1541_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14945)) := by
  have c94512 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94513 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94514 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94515 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94516 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94517 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_0 := arithEq_of_rows h (row := 1542) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_0
  simp only [← c94512, ← c94513, ← c94514] at e_1542_0
  have e_1542_1 := arithEq_of_rows h (row := 1542) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_1
  simp only [← c94515, ← c94516, k2, ← c94517] at e_1542_1
  have hr := e_1542_1
  simp only [e_1542_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14946)) := by
  have c94518 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94519 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94520 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94521 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94522 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94523 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_2 := arithEq_of_rows h (row := 1542) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_2
  simp only [← c94518, ← c94519, ← c94520] at e_1542_2
  have e_1542_3 := arithEq_of_rows h (row := 1542) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_3
  simp only [← c94521, ← c94522, k2, ← c94523] at e_1542_3
  have hr := e_1542_3
  simp only [e_1542_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14947)) := by
  have c94524 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94525 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94526 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94527 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94528 := (publicBatchWrapper89_copies2954 a h).1
  have c94529 := (publicBatchWrapper89_copies2954 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_4 := arithEq_of_rows h (row := 1542) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_4
  simp only [← c94524, ← c94525, ← c94526] at e_1542_4
  have e_1542_5 := arithEq_of_rows h (row := 1542) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_5
  simp only [← c94527, ← c94528, k2, ← c94529] at e_1542_5
  have hr := e_1542_5
  simp only [e_1542_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14948)) := by
  have c94530 := (publicBatchWrapper89_copies2954 a h).2.2.1
  have c94531 := (publicBatchWrapper89_copies2954 a h).2.2.2.1
  have c94532 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.1
  have c94533 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.1
  have c94534 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.1
  have c94535 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_6 := arithEq_of_rows h (row := 1542) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_6
  simp only [← c94530, ← c94531, ← c94532] at e_1542_6
  have e_1542_7 := arithEq_of_rows h (row := 1542) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_7
  simp only [← c94533, ← c94534, k2, ← c94535] at e_1542_7
  have hr := e_1542_7
  simp only [e_1542_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14949)) := by
  have c94536 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.1
  have c94537 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.1
  have c94538 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94539 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94540 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94541 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_8 := arithEq_of_rows h (row := 1542) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_8
  simp only [← c94536, ← c94537, ← c94538] at e_1542_8
  have e_1542_9 := arithEq_of_rows h (row := 1542) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_9
  simp only [← c94539, ← c94540, k2, ← c94541] at e_1542_9
  have hr := e_1542_9
  simp only [e_1542_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14950)) := by
  have c94542 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94543 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94544 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94545 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94546 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94547 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_10 := arithEq_of_rows h (row := 1542) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_10
  simp only [← c94542, ← c94543, ← c94544] at e_1542_10
  have e_1542_11 := arithEq_of_rows h (row := 1542) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_11
  simp only [← c94545, ← c94546, k2, ← c94547] at e_1542_11
  have hr := e_1542_11
  simp only [e_1542_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14951)) := by
  have c94548 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94549 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94550 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94551 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94552 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94553 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_12 := arithEq_of_rows h (row := 1542) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_12
  simp only [← c94548, ← c94549, ← c94550] at e_1542_12
  have e_1542_13 := arithEq_of_rows h (row := 1542) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_13
  simp only [← c94551, ← c94552, k2, ← c94553] at e_1542_13
  have hr := e_1542_13
  simp only [e_1542_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14952)) := by
  have c94554 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94555 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94556 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94557 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94558 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94559 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_14 := arithEq_of_rows h (row := 1542) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_14
  simp only [← c94554, ← c94555, ← c94556] at e_1542_14
  have e_1542_15 := arithEq_of_rows h (row := 1542) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_15
  simp only [← c94557, ← c94558, k2, ← c94559] at e_1542_15
  have hr := e_1542_15
  simp only [e_1542_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14953)) := by
  have c94560 := (publicBatchWrapper89_copies2955 a h).1
  have c94561 := (publicBatchWrapper89_copies2955 a h).2.1
  have c94562 := (publicBatchWrapper89_copies2955 a h).2.2.1
  have c94563 := (publicBatchWrapper89_copies2955 a h).2.2.2.1
  have c94564 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.1
  have c94565 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_16 := arithEq_of_rows h (row := 1542) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_16
  simp only [← c94560, ← c94561, ← c94562] at e_1542_16
  have e_1542_17 := arithEq_of_rows h (row := 1542) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_17
  simp only [← c94563, ← c94564, k2, ← c94565] at e_1542_17
  have hr := e_1542_17
  simp only [e_1542_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14954)) := by
  have c94566 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.1
  have c94567 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.1
  have c94568 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.1
  have c94569 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.1
  have c94570 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94571 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_18 := arithEq_of_rows h (row := 1542) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_18
  simp only [← c94566, ← c94567, ← c94568] at e_1542_18
  have e_1542_19 := arithEq_of_rows h (row := 1542) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_19
  simp only [← c94569, ← c94570, k2, ← c94571] at e_1542_19
  have hr := e_1542_19
  simp only [e_1542_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14955)) := by
  have c94572 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94573 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94574 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94575 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94576 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94577 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_0 := arithEq_of_rows h (row := 1543) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_0
  simp only [← c94572, ← c94573, ← c94574] at e_1543_0
  have e_1543_1 := arithEq_of_rows h (row := 1543) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_1
  simp only [← c94575, ← c94576, k2, ← c94577] at e_1543_1
  have hr := e_1543_1
  simp only [e_1543_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14956)) := by
  have c94578 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94579 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94580 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94581 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94582 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94583 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_2 := arithEq_of_rows h (row := 1543) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_2
  simp only [← c94578, ← c94579, ← c94580] at e_1543_2
  have e_1543_3 := arithEq_of_rows h (row := 1543) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_3
  simp only [← c94581, ← c94582, k2, ← c94583] at e_1543_3
  have hr := e_1543_3
  simp only [e_1543_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14957)) := by
  have c94584 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94585 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94586 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94587 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94588 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94589 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_4 := arithEq_of_rows h (row := 1543) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_4
  simp only [← c94584, ← c94585, ← c94586] at e_1543_4
  have e_1543_5 := arithEq_of_rows h (row := 1543) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_5
  simp only [← c94587, ← c94588, k2, ← c94589] at e_1543_5
  have hr := e_1543_5
  simp only [e_1543_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14958)) := by
  have c94590 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94591 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94592 := (publicBatchWrapper89_copies2956 a h).1
  have c94593 := (publicBatchWrapper89_copies2956 a h).2.1
  have c94594 := (publicBatchWrapper89_copies2956 a h).2.2.1
  have c94595 := (publicBatchWrapper89_copies2956 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_6 := arithEq_of_rows h (row := 1543) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_6
  simp only [← c94590, ← c94591, ← c94592] at e_1543_6
  have e_1543_7 := arithEq_of_rows h (row := 1543) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_7
  simp only [← c94593, ← c94594, k2, ← c94595] at e_1543_7
  have hr := e_1543_7
  simp only [e_1543_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14959)) := by
  have c94596 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.1
  have c94597 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.1
  have c94598 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.1
  have c94599 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.1
  have c94600 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.1
  have c94601 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_8 := arithEq_of_rows h (row := 1543) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_8
  simp only [← c94596, ← c94597, ← c94598] at e_1543_8
  have e_1543_9 := arithEq_of_rows h (row := 1543) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_9
  simp only [← c94599, ← c94600, k2, ← c94601] at e_1543_9
  have hr := e_1543_9
  simp only [e_1543_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14960)) := by
  have c94602 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94603 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94604 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94605 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94606 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94607 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_10 := arithEq_of_rows h (row := 1543) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_10
  simp only [← c94602, ← c94603, ← c94604] at e_1543_10
  have e_1543_11 := arithEq_of_rows h (row := 1543) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_11
  simp only [← c94605, ← c94606, k2, ← c94607] at e_1543_11
  have hr := e_1543_11
  simp only [e_1543_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14961)) := by
  have c94608 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94609 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94610 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94611 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94612 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94613 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_12 := arithEq_of_rows h (row := 1543) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_12
  simp only [← c94608, ← c94609, ← c94610] at e_1543_12
  have e_1543_13 := arithEq_of_rows h (row := 1543) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_13
  simp only [← c94611, ← c94612, k2, ← c94613] at e_1543_13
  have hr := e_1543_13
  simp only [e_1543_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14962)) := by
  have c94614 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94615 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94616 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94617 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94618 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94619 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_14 := arithEq_of_rows h (row := 1543) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_14
  simp only [← c94614, ← c94615, ← c94616] at e_1543_14
  have e_1543_15 := arithEq_of_rows h (row := 1543) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_15
  simp only [← c94617, ← c94618, k2, ← c94619] at e_1543_15
  have hr := e_1543_15
  simp only [e_1543_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14963)) := by
  have c94620 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94621 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94622 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94623 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94624 := (publicBatchWrapper89_copies2957 a h).1
  have c94625 := (publicBatchWrapper89_copies2957 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_16 := arithEq_of_rows h (row := 1543) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_16
  simp only [← c94620, ← c94621, ← c94622] at e_1543_16
  have e_1543_17 := arithEq_of_rows h (row := 1543) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_17
  simp only [← c94623, ← c94624, k2, ← c94625] at e_1543_17
  have hr := e_1543_17
  simp only [e_1543_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14964)) := by
  have c94626 := (publicBatchWrapper89_copies2957 a h).2.2.1
  have c94627 := (publicBatchWrapper89_copies2957 a h).2.2.2.1
  have c94628 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.1
  have c94629 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.1
  have c94630 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.1
  have c94631 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_18 := arithEq_of_rows h (row := 1543) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_18
  simp only [← c94626, ← c94627, ← c94628] at e_1543_18
  have e_1543_19 := arithEq_of_rows h (row := 1543) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_19
  simp only [← c94629, ← c94630, k2, ← c94631] at e_1543_19
  have hr := e_1543_19
  simp only [e_1543_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14965)) := by
  have c94632 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.1
  have c94633 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.1
  have c94634 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94635 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94636 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94637 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_0 := arithEq_of_rows h (row := 1544) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_0
  simp only [← c94632, ← c94633, ← c94634] at e_1544_0
  have e_1544_1 := arithEq_of_rows h (row := 1544) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_1
  simp only [← c94635, ← c94636, k2, ← c94637] at e_1544_1
  have hr := e_1544_1
  simp only [e_1544_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14966)) := by
  have c94638 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94639 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94640 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94641 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94642 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94643 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_2 := arithEq_of_rows h (row := 1544) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_2
  simp only [← c94638, ← c94639, ← c94640] at e_1544_2
  have e_1544_3 := arithEq_of_rows h (row := 1544) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_3
  simp only [← c94641, ← c94642, k2, ← c94643] at e_1544_3
  have hr := e_1544_3
  simp only [e_1544_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14967)) := by
  have c94644 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94645 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94646 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94647 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94648 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94649 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_4 := arithEq_of_rows h (row := 1544) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_4
  simp only [← c94644, ← c94645, ← c94646] at e_1544_4
  have e_1544_5 := arithEq_of_rows h (row := 1544) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_5
  simp only [← c94647, ← c94648, k2, ← c94649] at e_1544_5
  have hr := e_1544_5
  simp only [e_1544_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14968)) := by
  have c94650 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94651 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94652 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94653 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94654 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94655 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_6 := arithEq_of_rows h (row := 1544) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_6
  simp only [← c94650, ← c94651, ← c94652] at e_1544_6
  have e_1544_7 := arithEq_of_rows h (row := 1544) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_7
  simp only [← c94653, ← c94654, k2, ← c94655] at e_1544_7
  have hr := e_1544_7
  simp only [e_1544_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14969)) := by
  have c94656 := (publicBatchWrapper89_copies2958 a h).1
  have c94657 := (publicBatchWrapper89_copies2958 a h).2.1
  have c94658 := (publicBatchWrapper89_copies2958 a h).2.2.1
  have c94659 := (publicBatchWrapper89_copies2958 a h).2.2.2.1
  have c94660 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.1
  have c94661 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_8 := arithEq_of_rows h (row := 1544) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_8
  simp only [← c94656, ← c94657, ← c94658] at e_1544_8
  have e_1544_9 := arithEq_of_rows h (row := 1544) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_9
  simp only [← c94659, ← c94660, k2, ← c94661] at e_1544_9
  have hr := e_1544_9
  simp only [e_1544_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15010)) := by
  have c94662 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.1
  have c94663 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.1
  have c94664 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.1
  have c94665 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.1
  have c94666 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94667 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_10 := arithEq_of_rows h (row := 1544) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_10
  simp only [← c94662, ← c94663, ← c94664] at e_1544_10
  have e_1544_11 := arithEq_of_rows h (row := 1544) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_11
  simp only [← c94665, ← c94666, k2, ← c94667] at e_1544_11
  have hr := e_1544_11
  simp only [e_1544_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15011)) := by
  have c94668 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94669 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94670 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94671 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94672 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94673 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_12 := arithEq_of_rows h (row := 1544) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_12
  simp only [← c94668, ← c94669, ← c94670] at e_1544_12
  have e_1544_13 := arithEq_of_rows h (row := 1544) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_13
  simp only [← c94671, ← c94672, k2, ← c94673] at e_1544_13
  have hr := e_1544_13
  simp only [e_1544_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15012)) := by
  have c94674 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94675 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94676 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94677 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94678 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94679 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_14 := arithEq_of_rows h (row := 1544) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_14
  simp only [← c94674, ← c94675, ← c94676] at e_1544_14
  have e_1544_15 := arithEq_of_rows h (row := 1544) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_15
  simp only [← c94677, ← c94678, k2, ← c94679] at e_1544_15
  have hr := e_1544_15
  simp only [e_1544_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15013)) := by
  have c94680 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94681 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94682 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94683 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94684 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94685 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_16 := arithEq_of_rows h (row := 1544) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_16
  simp only [← c94680, ← c94681, ← c94682] at e_1544_16
  have e_1544_17 := arithEq_of_rows h (row := 1544) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_17
  simp only [← c94683, ← c94684, k2, ← c94685] at e_1544_17
  have hr := e_1544_17
  simp only [e_1544_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15014)) := by
  have c94686 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94687 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94688 := (publicBatchWrapper89_copies2959 a h).1
  have c94689 := (publicBatchWrapper89_copies2959 a h).2.1
  have c94690 := (publicBatchWrapper89_copies2959 a h).2.2.1
  have c94691 := (publicBatchWrapper89_copies2959 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_18 := arithEq_of_rows h (row := 1544) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_18
  simp only [← c94686, ← c94687, ← c94688] at e_1544_18
  have e_1544_19 := arithEq_of_rows h (row := 1544) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_19
  simp only [← c94689, ← c94690, k2, ← c94691] at e_1544_19
  have hr := e_1544_19
  simp only [e_1544_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15015)) := by
  have c94692 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.1
  have c94693 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.1
  have c94694 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.1
  have c94695 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.1
  have c94696 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.1
  have c94697 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_0 := arithEq_of_rows h (row := 1545) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_0
  simp only [← c94692, ← c94693, ← c94694] at e_1545_0
  have e_1545_1 := arithEq_of_rows h (row := 1545) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_1
  simp only [← c94695, ← c94696, k2, ← c94697] at e_1545_1
  have hr := e_1545_1
  simp only [e_1545_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15016)) := by
  have c94698 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94699 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94700 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94701 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94702 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94703 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_2 := arithEq_of_rows h (row := 1545) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_2
  simp only [← c94698, ← c94699, ← c94700] at e_1545_2
  have e_1545_3 := arithEq_of_rows h (row := 1545) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_3
  simp only [← c94701, ← c94702, k2, ← c94703] at e_1545_3
  have hr := e_1545_3
  simp only [e_1545_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15017)) := by
  have c94704 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94705 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94706 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94707 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94708 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94709 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_4 := arithEq_of_rows h (row := 1545) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_4
  simp only [← c94704, ← c94705, ← c94706] at e_1545_4
  have e_1545_5 := arithEq_of_rows h (row := 1545) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_5
  simp only [← c94707, ← c94708, k2, ← c94709] at e_1545_5
  have hr := e_1545_5
  simp only [e_1545_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15018)) := by
  have c94710 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94711 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94712 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94713 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94714 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94715 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_6 := arithEq_of_rows h (row := 1545) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_6
  simp only [← c94710, ← c94711, ← c94712] at e_1545_6
  have e_1545_7 := arithEq_of_rows h (row := 1545) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_7
  simp only [← c94713, ← c94714, k2, ← c94715] at e_1545_7
  have hr := e_1545_7
  simp only [e_1545_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15019)) := by
  have c94716 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94717 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94718 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94719 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94720 := (publicBatchWrapper89_copies2960 a h).1
  have c94721 := (publicBatchWrapper89_copies2960 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_8 := arithEq_of_rows h (row := 1545) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_8
  simp only [← c94716, ← c94717, ← c94718] at e_1545_8
  have e_1545_9 := arithEq_of_rows h (row := 1545) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_9
  simp only [← c94719, ← c94720, k2, ← c94721] at e_1545_9
  have hr := e_1545_9
  simp only [e_1545_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15020)) := by
  have c94722 := (publicBatchWrapper89_copies2960 a h).2.2.1
  have c94723 := (publicBatchWrapper89_copies2960 a h).2.2.2.1
  have c94724 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.1
  have c94725 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.1
  have c94726 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.1
  have c94727 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_10 := arithEq_of_rows h (row := 1545) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_10
  simp only [← c94722, ← c94723, ← c94724] at e_1545_10
  have e_1545_11 := arithEq_of_rows h (row := 1545) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_11
  simp only [← c94725, ← c94726, k2, ← c94727] at e_1545_11
  have hr := e_1545_11
  simp only [e_1545_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15021)) := by
  have c94728 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.1
  have c94729 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.1
  have c94730 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94731 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94732 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94733 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_12 := arithEq_of_rows h (row := 1545) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_12
  simp only [← c94728, ← c94729, ← c94730] at e_1545_12
  have e_1545_13 := arithEq_of_rows h (row := 1545) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_13
  simp only [← c94731, ← c94732, k2, ← c94733] at e_1545_13
  have hr := e_1545_13
  simp only [e_1545_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15022)) := by
  have c94734 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94735 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94736 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94737 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94738 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94739 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_14 := arithEq_of_rows h (row := 1545) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_14
  simp only [← c94734, ← c94735, ← c94736] at e_1545_14
  have e_1545_15 := arithEq_of_rows h (row := 1545) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_15
  simp only [← c94737, ← c94738, k2, ← c94739] at e_1545_15
  have hr := e_1545_15
  simp only [e_1545_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15023)) := by
  have c94740 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94741 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94742 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94743 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94744 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94745 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_16 := arithEq_of_rows h (row := 1545) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_16
  simp only [← c94740, ← c94741, ← c94742] at e_1545_16
  have e_1545_17 := arithEq_of_rows h (row := 1545) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_17
  simp only [← c94743, ← c94744, k2, ← c94745] at e_1545_17
  have hr := e_1545_17
  simp only [e_1545_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15024)) := by
  have c94746 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94747 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94748 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94749 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94750 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94751 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_18 := arithEq_of_rows h (row := 1545) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_18
  simp only [← c94746, ← c94747, ← c94748] at e_1545_18
  have e_1545_19 := arithEq_of_rows h (row := 1545) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_19
  simp only [← c94749, ← c94750, k2, ← c94751] at e_1545_19
  have hr := e_1545_19
  simp only [e_1545_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15025)) := by
  have c94752 := (publicBatchWrapper89_copies2961 a h).1
  have c94753 := (publicBatchWrapper89_copies2961 a h).2.1
  have c94754 := (publicBatchWrapper89_copies2961 a h).2.2.1
  have c94755 := (publicBatchWrapper89_copies2961 a h).2.2.2.1
  have c94756 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.1
  have c94757 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_0 := arithEq_of_rows h (row := 1546) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_0
  simp only [← c94752, ← c94753, ← c94754] at e_1546_0
  have e_1546_1 := arithEq_of_rows h (row := 1546) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_1
  simp only [← c94755, ← c94756, k2, ← c94757] at e_1546_1
  have hr := e_1546_1
  simp only [e_1546_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15026)) := by
  have c94758 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.1
  have c94759 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.1
  have c94760 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.1
  have c94761 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.1
  have c94762 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94763 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_2 := arithEq_of_rows h (row := 1546) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_2
  simp only [← c94758, ← c94759, ← c94760] at e_1546_2
  have e_1546_3 := arithEq_of_rows h (row := 1546) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_3
  simp only [← c94761, ← c94762, k2, ← c94763] at e_1546_3
  have hr := e_1546_3
  simp only [e_1546_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15027)) := by
  have c94764 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94765 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94766 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94767 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94768 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94769 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_4 := arithEq_of_rows h (row := 1546) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_4
  simp only [← c94764, ← c94765, ← c94766] at e_1546_4
  have e_1546_5 := arithEq_of_rows h (row := 1546) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_5
  simp only [← c94767, ← c94768, k2, ← c94769] at e_1546_5
  have hr := e_1546_5
  simp only [e_1546_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15028)) := by
  have c94770 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94771 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94772 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94773 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94774 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94775 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_6 := arithEq_of_rows h (row := 1546) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_6
  simp only [← c94770, ← c94771, ← c94772] at e_1546_6
  have e_1546_7 := arithEq_of_rows h (row := 1546) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_7
  simp only [← c94773, ← c94774, k2, ← c94775] at e_1546_7
  have hr := e_1546_7
  simp only [e_1546_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15029)) := by
  have c94776 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94777 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94778 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94779 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94780 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94781 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_8 := arithEq_of_rows h (row := 1546) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_8
  simp only [← c94776, ← c94777, ← c94778] at e_1546_8
  have e_1546_9 := arithEq_of_rows h (row := 1546) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_9
  simp only [← c94779, ← c94780, k2, ← c94781] at e_1546_9
  have hr := e_1546_9
  simp only [e_1546_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15030)) := by
  have c94782 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94783 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94784 := (publicBatchWrapper89_copies2962 a h).1
  have c94785 := (publicBatchWrapper89_copies2962 a h).2.1
  have c94786 := (publicBatchWrapper89_copies2962 a h).2.2.1
  have c94787 := (publicBatchWrapper89_copies2962 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_10 := arithEq_of_rows h (row := 1546) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_10
  simp only [← c94782, ← c94783, ← c94784] at e_1546_10
  have e_1546_11 := arithEq_of_rows h (row := 1546) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_11
  simp only [← c94785, ← c94786, k2, ← c94787] at e_1546_11
  have hr := e_1546_11
  simp only [e_1546_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15031)) := by
  have c94788 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.1
  have c94789 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.1
  have c94790 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.1
  have c94791 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.1
  have c94792 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.1
  have c94793 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_12 := arithEq_of_rows h (row := 1546) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_12
  simp only [← c94788, ← c94789, ← c94790] at e_1546_12
  have e_1546_13 := arithEq_of_rows h (row := 1546) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_13
  simp only [← c94791, ← c94792, k2, ← c94793] at e_1546_13
  have hr := e_1546_13
  simp only [e_1546_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15032)) := by
  have c94794 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94795 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94796 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94797 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94798 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94799 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_14 := arithEq_of_rows h (row := 1546) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_14
  simp only [← c94794, ← c94795, ← c94796] at e_1546_14
  have e_1546_15 := arithEq_of_rows h (row := 1546) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_15
  simp only [← c94797, ← c94798, k2, ← c94799] at e_1546_15
  have hr := e_1546_15
  simp only [e_1546_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15033)) := by
  have c94800 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94801 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94802 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94803 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94804 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94805 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_16 := arithEq_of_rows h (row := 1546) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_16
  simp only [← c94800, ← c94801, ← c94802] at e_1546_16
  have e_1546_17 := arithEq_of_rows h (row := 1546) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_17
  simp only [← c94803, ← c94804, k2, ← c94805] at e_1546_17
  have hr := e_1546_17
  simp only [e_1546_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15034)) := by
  have c94806 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94807 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94808 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94809 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94810 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94811 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_18 := arithEq_of_rows h (row := 1546) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_18
  simp only [← c94806, ← c94807, ← c94808] at e_1546_18
  have e_1546_19 := arithEq_of_rows h (row := 1546) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_19
  simp only [← c94809, ← c94810, k2, ← c94811] at e_1546_19
  have hr := e_1546_19
  simp only [e_1546_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15035)) := by
  have c94812 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94813 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94814 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94815 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94816 := (publicBatchWrapper89_copies2963 a h).1
  have c94817 := (publicBatchWrapper89_copies2963 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_0 := arithEq_of_rows h (row := 1547) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_0
  simp only [← c94812, ← c94813, ← c94814] at e_1547_0
  have e_1547_1 := arithEq_of_rows h (row := 1547) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_1
  simp only [← c94815, ← c94816, k2, ← c94817] at e_1547_1
  have hr := e_1547_1
  simp only [e_1547_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15036)) := by
  have c94818 := (publicBatchWrapper89_copies2963 a h).2.2.1
  have c94819 := (publicBatchWrapper89_copies2963 a h).2.2.2.1
  have c94820 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.1
  have c94821 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.1
  have c94822 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.1
  have c94823 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_2 := arithEq_of_rows h (row := 1547) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_2
  simp only [← c94818, ← c94819, ← c94820] at e_1547_2
  have e_1547_3 := arithEq_of_rows h (row := 1547) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_3
  simp only [← c94821, ← c94822, k2, ← c94823] at e_1547_3
  have hr := e_1547_3
  simp only [e_1547_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15037)) := by
  have c94824 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.1
  have c94825 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.1
  have c94826 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94827 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94828 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94829 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_4 := arithEq_of_rows h (row := 1547) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_4
  simp only [← c94824, ← c94825, ← c94826] at e_1547_4
  have e_1547_5 := arithEq_of_rows h (row := 1547) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_5
  simp only [← c94827, ← c94828, k2, ← c94829] at e_1547_5
  have hr := e_1547_5
  simp only [e_1547_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15038)) := by
  have c94830 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94831 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94832 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94833 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94834 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94835 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_6 := arithEq_of_rows h (row := 1547) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_6
  simp only [← c94830, ← c94831, ← c94832] at e_1547_6
  have e_1547_7 := arithEq_of_rows h (row := 1547) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_7
  simp only [← c94833, ← c94834, k2, ← c94835] at e_1547_7
  have hr := e_1547_7
  simp only [e_1547_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15039)) := by
  have c94836 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94837 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94838 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94839 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94840 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94841 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_8 := arithEq_of_rows h (row := 1547) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_8
  simp only [← c94836, ← c94837, ← c94838] at e_1547_8
  have e_1547_9 := arithEq_of_rows h (row := 1547) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_9
  simp only [← c94839, ← c94840, k2, ← c94841] at e_1547_9
  have hr := e_1547_9
  simp only [e_1547_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15040)) := by
  have c94842 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94843 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94844 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94845 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94846 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94847 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_10 := arithEq_of_rows h (row := 1547) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_10
  simp only [← c94842, ← c94843, ← c94844] at e_1547_10
  have e_1547_11 := arithEq_of_rows h (row := 1547) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_11
  simp only [← c94845, ← c94846, k2, ← c94847] at e_1547_11
  have hr := e_1547_11
  simp only [e_1547_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15041)) := by
  have c94848 := (publicBatchWrapper89_copies2964 a h).1
  have c94849 := (publicBatchWrapper89_copies2964 a h).2.1
  have c94850 := (publicBatchWrapper89_copies2964 a h).2.2.1
  have c94851 := (publicBatchWrapper89_copies2964 a h).2.2.2.1
  have c94852 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.1
  have c94853 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_12 := arithEq_of_rows h (row := 1547) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_12
  simp only [← c94848, ← c94849, ← c94850] at e_1547_12
  have e_1547_13 := arithEq_of_rows h (row := 1547) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_13
  simp only [← c94851, ← c94852, k2, ← c94853] at e_1547_13
  have hr := e_1547_13
  simp only [e_1547_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15042)) := by
  have c94854 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.1
  have c94855 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.1
  have c94856 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.1
  have c94857 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.1
  have c94858 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94859 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_14 := arithEq_of_rows h (row := 1547) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_14
  simp only [← c94854, ← c94855, ← c94856] at e_1547_14
  have e_1547_15 := arithEq_of_rows h (row := 1547) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_15
  simp only [← c94857, ← c94858, k2, ← c94859] at e_1547_15
  have hr := e_1547_15
  simp only [e_1547_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15043)) := by
  have c94860 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94861 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94862 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94863 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94864 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94865 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_16 := arithEq_of_rows h (row := 1547) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_16
  simp only [← c94860, ← c94861, ← c94862] at e_1547_16
  have e_1547_17 := arithEq_of_rows h (row := 1547) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_17
  simp only [← c94863, ← c94864, k2, ← c94865] at e_1547_17
  have hr := e_1547_17
  simp only [e_1547_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15044)) := by
  have c94866 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94867 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94868 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94869 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94870 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94871 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_18 := arithEq_of_rows h (row := 1547) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_18
  simp only [← c94866, ← c94867, ← c94868] at e_1547_18
  have e_1547_19 := arithEq_of_rows h (row := 1547) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_19
  simp only [← c94869, ← c94870, k2, ← c94871] at e_1547_19
  have hr := e_1547_19
  simp only [e_1547_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15045)) := by
  have c94872 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94873 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94874 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94875 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94876 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94877 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_0 := arithEq_of_rows h (row := 1548) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_0
  simp only [← c94872, ← c94873, ← c94874] at e_1548_0
  have e_1548_1 := arithEq_of_rows h (row := 1548) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_1
  simp only [← c94875, ← c94876, k2, ← c94877] at e_1548_1
  have hr := e_1548_1
  simp only [e_1548_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15046)) := by
  have c94878 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94879 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94880 := (publicBatchWrapper89_copies2965 a h).1
  have c94881 := (publicBatchWrapper89_copies2965 a h).2.1
  have c94882 := (publicBatchWrapper89_copies2965 a h).2.2.1
  have c94883 := (publicBatchWrapper89_copies2965 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_2 := arithEq_of_rows h (row := 1548) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_2
  simp only [← c94878, ← c94879, ← c94880] at e_1548_2
  have e_1548_3 := arithEq_of_rows h (row := 1548) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_3
  simp only [← c94881, ← c94882, k2, ← c94883] at e_1548_3
  have hr := e_1548_3
  simp only [e_1548_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15047)) := by
  have c94884 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.1
  have c94885 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.1
  have c94886 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.1
  have c94887 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.1
  have c94888 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.1
  have c94889 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_4 := arithEq_of_rows h (row := 1548) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_4
  simp only [← c94884, ← c94885, ← c94886] at e_1548_4
  have e_1548_5 := arithEq_of_rows h (row := 1548) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_5
  simp only [← c94887, ← c94888, k2, ← c94889] at e_1548_5
  have hr := e_1548_5
  simp only [e_1548_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15048)) := by
  have c94890 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94891 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94892 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94893 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94894 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94895 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_6 := arithEq_of_rows h (row := 1548) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_6
  simp only [← c94890, ← c94891, ← c94892] at e_1548_6
  have e_1548_7 := arithEq_of_rows h (row := 1548) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_7
  simp only [← c94893, ← c94894, k2, ← c94895] at e_1548_7
  have hr := e_1548_7
  simp only [e_1548_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15049)) := by
  have c94896 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94897 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94898 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94899 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94900 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94901 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_8 := arithEq_of_rows h (row := 1548) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_8
  simp only [← c94896, ← c94897, ← c94898] at e_1548_8
  have e_1548_9 := arithEq_of_rows h (row := 1548) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_9
  simp only [← c94899, ← c94900, k2, ← c94901] at e_1548_9
  have hr := e_1548_9
  simp only [e_1548_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15050)) := by
  have c94902 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94903 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94904 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94905 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94906 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94907 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_10 := arithEq_of_rows h (row := 1548) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_10
  simp only [← c94902, ← c94903, ← c94904] at e_1548_10
  have e_1548_11 := arithEq_of_rows h (row := 1548) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_11
  simp only [← c94905, ← c94906, k2, ← c94907] at e_1548_11
  have hr := e_1548_11
  simp only [e_1548_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15051)) := by
  have c94908 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94909 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94910 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94911 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94912 := (publicBatchWrapper89_copies2966 a h).1
  have c94913 := (publicBatchWrapper89_copies2966 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_12 := arithEq_of_rows h (row := 1548) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_12
  simp only [← c94908, ← c94909, ← c94910] at e_1548_12
  have e_1548_13 := arithEq_of_rows h (row := 1548) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_13
  simp only [← c94911, ← c94912, k2, ← c94913] at e_1548_13
  have hr := e_1548_13
  simp only [e_1548_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15052)) := by
  have c94914 := (publicBatchWrapper89_copies2966 a h).2.2.1
  have c94915 := (publicBatchWrapper89_copies2966 a h).2.2.2.1
  have c94916 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.1
  have c94917 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.1
  have c94918 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.1
  have c94919 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_14 := arithEq_of_rows h (row := 1548) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_14
  simp only [← c94914, ← c94915, ← c94916] at e_1548_14
  have e_1548_15 := arithEq_of_rows h (row := 1548) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_15
  simp only [← c94917, ← c94918, k2, ← c94919] at e_1548_15
  have hr := e_1548_15
  simp only [e_1548_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15053)) := by
  have c94920 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.1
  have c94921 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.1
  have c94922 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94923 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94924 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94925 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_16 := arithEq_of_rows h (row := 1548) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_16
  simp only [← c94920, ← c94921, ← c94922] at e_1548_16
  have e_1548_17 := arithEq_of_rows h (row := 1548) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_17
  simp only [← c94923, ← c94924, k2, ← c94925] at e_1548_17
  have hr := e_1548_17
  simp only [e_1548_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15054)) := by
  have c94926 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94927 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94928 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94929 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94930 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94931 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_18 := arithEq_of_rows h (row := 1548) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_18
  simp only [← c94926, ← c94927, ← c94928] at e_1548_18
  have e_1548_19 := arithEq_of_rows h (row := 1548) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_19
  simp only [← c94929, ← c94930, k2, ← c94931] at e_1548_19
  have hr := e_1548_19
  simp only [e_1548_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15055)) := by
  have c94932 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94933 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94934 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94935 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94936 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94937 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_0 := arithEq_of_rows h (row := 1549) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_0
  simp only [← c94932, ← c94933, ← c94934] at e_1549_0
  have e_1549_1 := arithEq_of_rows h (row := 1549) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_1
  simp only [← c94935, ← c94936, k2, ← c94937] at e_1549_1
  have hr := e_1549_1
  simp only [e_1549_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15056)) := by
  have c94938 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94939 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94940 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94941 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94942 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94943 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_2 := arithEq_of_rows h (row := 1549) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_2
  simp only [← c94938, ← c94939, ← c94940] at e_1549_2
  have e_1549_3 := arithEq_of_rows h (row := 1549) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_3
  simp only [← c94941, ← c94942, k2, ← c94943] at e_1549_3
  have hr := e_1549_3
  simp only [e_1549_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15057)) := by
  have c94944 := (publicBatchWrapper89_copies2967 a h).1
  have c94945 := (publicBatchWrapper89_copies2967 a h).2.1
  have c94946 := (publicBatchWrapper89_copies2967 a h).2.2.1
  have c94947 := (publicBatchWrapper89_copies2967 a h).2.2.2.1
  have c94948 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.1
  have c94949 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_4 := arithEq_of_rows h (row := 1549) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_4
  simp only [← c94944, ← c94945, ← c94946] at e_1549_4
  have e_1549_5 := arithEq_of_rows h (row := 1549) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_5
  simp only [← c94947, ← c94948, k2, ← c94949] at e_1549_5
  have hr := e_1549_5
  simp only [e_1549_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15058)) := by
  have c94950 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.1
  have c94951 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.1
  have c94952 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.1
  have c94953 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.1
  have c94954 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94955 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_6 := arithEq_of_rows h (row := 1549) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_6
  simp only [← c94950, ← c94951, ← c94952] at e_1549_6
  have e_1549_7 := arithEq_of_rows h (row := 1549) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_7
  simp only [← c94953, ← c94954, k2, ← c94955] at e_1549_7
  have hr := e_1549_7
  simp only [e_1549_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15059)) := by
  have c94956 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94957 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94958 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94959 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94960 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94961 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_8 := arithEq_of_rows h (row := 1549) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_8
  simp only [← c94956, ← c94957, ← c94958] at e_1549_8
  have e_1549_9 := arithEq_of_rows h (row := 1549) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_9
  simp only [← c94959, ← c94960, k2, ← c94961] at e_1549_9
  have hr := e_1549_9
  simp only [e_1549_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15060)) := by
  have c94962 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94963 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94964 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94965 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94966 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94967 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_10 := arithEq_of_rows h (row := 1549) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_10
  simp only [← c94962, ← c94963, ← c94964] at e_1549_10
  have e_1549_11 := arithEq_of_rows h (row := 1549) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_11
  simp only [← c94965, ← c94966, k2, ← c94967] at e_1549_11
  have hr := e_1549_11
  simp only [e_1549_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15061)) := by
  have c94968 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94969 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94970 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94971 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94972 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94973 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_12 := arithEq_of_rows h (row := 1549) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_12
  simp only [← c94968, ← c94969, ← c94970] at e_1549_12
  have e_1549_13 := arithEq_of_rows h (row := 1549) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_13
  simp only [← c94971, ← c94972, k2, ← c94973] at e_1549_13
  have hr := e_1549_13
  simp only [e_1549_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15062)) := by
  have c94974 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94975 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94976 := (publicBatchWrapper89_copies2968 a h).1
  have c94977 := (publicBatchWrapper89_copies2968 a h).2.1
  have c94978 := (publicBatchWrapper89_copies2968 a h).2.2.1
  have c94979 := (publicBatchWrapper89_copies2968 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_14 := arithEq_of_rows h (row := 1549) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_14
  simp only [← c94974, ← c94975, ← c94976] at e_1549_14
  have e_1549_15 := arithEq_of_rows h (row := 1549) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_15
  simp only [← c94977, ← c94978, k2, ← c94979] at e_1549_15
  have hr := e_1549_15
  simp only [e_1549_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15063)) := by
  have c94980 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.1
  have c94981 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.1
  have c94982 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.1
  have c94983 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.1
  have c94984 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.1
  have c94985 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_16 := arithEq_of_rows h (row := 1549) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_16
  simp only [← c94980, ← c94981, ← c94982] at e_1549_16
  have e_1549_17 := arithEq_of_rows h (row := 1549) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_17
  simp only [← c94983, ← c94984, k2, ← c94985] at e_1549_17
  have hr := e_1549_17
  simp only [e_1549_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15064)) := by
  have c94986 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94987 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94988 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94989 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94990 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94991 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_18 := arithEq_of_rows h (row := 1549) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_18
  simp only [← c94986, ← c94987, ← c94988] at e_1549_18
  have e_1549_19 := arithEq_of_rows h (row := 1549) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_19
  simp only [← c94989, ← c94990, k2, ← c94991] at e_1549_19
  have hr := e_1549_19
  simp only [e_1549_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15065)) := by
  have c94992 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94993 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94994 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94995 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94996 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94997 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_0 := arithEq_of_rows h (row := 1550) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_0
  simp only [← c94992, ← c94993, ← c94994] at e_1550_0
  have e_1550_1 := arithEq_of_rows h (row := 1550) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_1
  simp only [← c94995, ← c94996, k2, ← c94997] at e_1550_1
  have hr := e_1550_1
  simp only [e_1550_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15066)) := by
  have c94998 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94999 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95000 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95001 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95002 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95003 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_2 := arithEq_of_rows h (row := 1550) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_2
  simp only [← c94998, ← c94999, ← c95000] at e_1550_2
  have e_1550_3 := arithEq_of_rows h (row := 1550) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_3
  simp only [← c95001, ← c95002, k2, ← c95003] at e_1550_3
  have hr := e_1550_3
  simp only [e_1550_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15067)) := by
  have c95004 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95005 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95006 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95007 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95008 := (publicBatchWrapper89_copies2969 a h).1
  have c95009 := (publicBatchWrapper89_copies2969 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_4 := arithEq_of_rows h (row := 1550) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_4
  simp only [← c95004, ← c95005, ← c95006] at e_1550_4
  have e_1550_5 := arithEq_of_rows h (row := 1550) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_5
  simp only [← c95007, ← c95008, k2, ← c95009] at e_1550_5
  have hr := e_1550_5
  simp only [e_1550_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15068)) := by
  have c95010 := (publicBatchWrapper89_copies2969 a h).2.2.1
  have c95011 := (publicBatchWrapper89_copies2969 a h).2.2.2.1
  have c95012 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.1
  have c95013 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.1
  have c95014 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.1
  have c95015 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_6 := arithEq_of_rows h (row := 1550) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_6
  simp only [← c95010, ← c95011, ← c95012] at e_1550_6
  have e_1550_7 := arithEq_of_rows h (row := 1550) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_7
  simp only [← c95013, ← c95014, k2, ← c95015] at e_1550_7
  have hr := e_1550_7
  simp only [e_1550_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15069)) := by
  have c95016 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.1
  have c95017 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.1
  have c95018 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95019 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95020 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95021 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_8 := arithEq_of_rows h (row := 1550) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_8
  simp only [← c95016, ← c95017, ← c95018] at e_1550_8
  have e_1550_9 := arithEq_of_rows h (row := 1550) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_9
  simp only [← c95019, ← c95020, k2, ← c95021] at e_1550_9
  have hr := e_1550_9
  simp only [e_1550_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15070)) := by
  have c95022 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95023 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95024 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95025 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95026 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95027 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_10 := arithEq_of_rows h (row := 1550) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_10
  simp only [← c95022, ← c95023, ← c95024] at e_1550_10
  have e_1550_11 := arithEq_of_rows h (row := 1550) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_11
  simp only [← c95025, ← c95026, k2, ← c95027] at e_1550_11
  have hr := e_1550_11
  simp only [e_1550_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15071)) := by
  have c95028 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95029 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95030 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95031 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95032 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95033 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_12 := arithEq_of_rows h (row := 1550) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_12
  simp only [← c95028, ← c95029, ← c95030] at e_1550_12
  have e_1550_13 := arithEq_of_rows h (row := 1550) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_13
  simp only [← c95031, ← c95032, k2, ← c95033] at e_1550_13
  have hr := e_1550_13
  simp only [e_1550_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15072)) := by
  have c95034 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95035 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95036 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95037 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95038 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95039 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_14 := arithEq_of_rows h (row := 1550) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_14
  simp only [← c95034, ← c95035, ← c95036] at e_1550_14
  have e_1550_15 := arithEq_of_rows h (row := 1550) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_15
  simp only [← c95037, ← c95038, k2, ← c95039] at e_1550_15
  have hr := e_1550_15
  simp only [e_1550_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15073)) := by
  have c95040 := (publicBatchWrapper89_copies2970 a h).1
  have c95041 := (publicBatchWrapper89_copies2970 a h).2.1
  have c95042 := (publicBatchWrapper89_copies2970 a h).2.2.1
  have c95043 := (publicBatchWrapper89_copies2970 a h).2.2.2.1
  have c95044 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.1
  have c95045 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_16 := arithEq_of_rows h (row := 1550) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_16
  simp only [← c95040, ← c95041, ← c95042] at e_1550_16
  have e_1550_17 := arithEq_of_rows h (row := 1550) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_17
  simp only [← c95043, ← c95044, k2, ← c95045] at e_1550_17
  have hr := e_1550_17
  simp only [e_1550_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15074)) := by
  have c95046 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.1
  have c95047 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.1
  have c95048 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.1
  have c95049 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.1
  have c95050 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95051 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_18 := arithEq_of_rows h (row := 1550) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_18
  simp only [← c95046, ← c95047, ← c95048] at e_1550_18
  have e_1550_19 := arithEq_of_rows h (row := 1550) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_19
  simp only [← c95049, ← c95050, k2, ← c95051] at e_1550_19
  have hr := e_1550_19
  simp only [e_1550_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15075)) := by
  have c95052 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95053 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95054 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95055 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95056 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95057 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_0 := arithEq_of_rows h (row := 1551) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_0
  simp only [← c95052, ← c95053, ← c95054] at e_1551_0
  have e_1551_1 := arithEq_of_rows h (row := 1551) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_1
  simp only [← c95055, ← c95056, k2, ← c95057] at e_1551_1
  have hr := e_1551_1
  simp only [e_1551_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15076)) := by
  have c95058 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95059 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95060 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95061 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95062 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95063 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_2 := arithEq_of_rows h (row := 1551) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_2
  simp only [← c95058, ← c95059, ← c95060] at e_1551_2
  have e_1551_3 := arithEq_of_rows h (row := 1551) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_3
  simp only [← c95061, ← c95062, k2, ← c95063] at e_1551_3
  have hr := e_1551_3
  simp only [e_1551_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15077)) := by
  have c95064 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95065 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95066 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95067 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95068 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95069 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_4 := arithEq_of_rows h (row := 1551) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_4
  simp only [← c95064, ← c95065, ← c95066] at e_1551_4
  have e_1551_5 := arithEq_of_rows h (row := 1551) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_5
  simp only [← c95067, ← c95068, k2, ← c95069] at e_1551_5
  have hr := e_1551_5
  simp only [e_1551_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15078)) := by
  have c95070 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95071 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95072 := (publicBatchWrapper89_copies2971 a h).1
  have c95073 := (publicBatchWrapper89_copies2971 a h).2.1
  have c95074 := (publicBatchWrapper89_copies2971 a h).2.2.1
  have c95075 := (publicBatchWrapper89_copies2971 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_6 := arithEq_of_rows h (row := 1551) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_6
  simp only [← c95070, ← c95071, ← c95072] at e_1551_6
  have e_1551_7 := arithEq_of_rows h (row := 1551) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_7
  simp only [← c95073, ← c95074, k2, ← c95075] at e_1551_7
  have hr := e_1551_7
  simp only [e_1551_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15079)) := by
  have c95076 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.1
  have c95077 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.1
  have c95078 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.1
  have c95079 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.1
  have c95080 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.1
  have c95081 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_8 := arithEq_of_rows h (row := 1551) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_8
  simp only [← c95076, ← c95077, ← c95078] at e_1551_8
  have e_1551_9 := arithEq_of_rows h (row := 1551) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_9
  simp only [← c95079, ← c95080, k2, ← c95081] at e_1551_9
  have hr := e_1551_9
  simp only [e_1551_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15080)) := by
  have c95082 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95083 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95084 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95085 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95086 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95087 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_10 := arithEq_of_rows h (row := 1551) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_10
  simp only [← c95082, ← c95083, ← c95084] at e_1551_10
  have e_1551_11 := arithEq_of_rows h (row := 1551) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_11
  simp only [← c95085, ← c95086, k2, ← c95087] at e_1551_11
  have hr := e_1551_11
  simp only [e_1551_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15081)) := by
  have c95088 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95089 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95090 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95091 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95092 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95093 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_12 := arithEq_of_rows h (row := 1551) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_12
  simp only [← c95088, ← c95089, ← c95090] at e_1551_12
  have e_1551_13 := arithEq_of_rows h (row := 1551) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_13
  simp only [← c95091, ← c95092, k2, ← c95093] at e_1551_13
  have hr := e_1551_13
  simp only [e_1551_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15082)) := by
  have c95094 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95095 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95096 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95097 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95098 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95099 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_14 := arithEq_of_rows h (row := 1551) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_14
  simp only [← c95094, ← c95095, ← c95096] at e_1551_14
  have e_1551_15 := arithEq_of_rows h (row := 1551) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_15
  simp only [← c95097, ← c95098, k2, ← c95099] at e_1551_15
  have hr := e_1551_15
  simp only [e_1551_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15083)) := by
  have c95100 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95101 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95102 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95103 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95104 := (publicBatchWrapper89_copies2972 a h).1
  have c95105 := (publicBatchWrapper89_copies2972 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_16 := arithEq_of_rows h (row := 1551) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_16
  simp only [← c95100, ← c95101, ← c95102] at e_1551_16
  have e_1551_17 := arithEq_of_rows h (row := 1551) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_17
  simp only [← c95103, ← c95104, k2, ← c95105] at e_1551_17
  have hr := e_1551_17
  simp only [e_1551_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15084)) := by
  have c95106 := (publicBatchWrapper89_copies2972 a h).2.2.1
  have c95107 := (publicBatchWrapper89_copies2972 a h).2.2.2.1
  have c95108 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.1
  have c95109 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.1
  have c95110 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.1
  have c95111 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_18 := arithEq_of_rows h (row := 1551) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_18
  simp only [← c95106, ← c95107, ← c95108] at e_1551_18
  have e_1551_19 := arithEq_of_rows h (row := 1551) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_19
  simp only [← c95109, ← c95110, k2, ← c95111] at e_1551_19
  have hr := e_1551_19
  simp only [e_1551_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15085)) := by
  have c95112 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.1
  have c95113 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.1
  have c95114 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95115 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95116 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95117 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_0 := arithEq_of_rows h (row := 1552) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_0
  simp only [← c95112, ← c95113, ← c95114] at e_1552_0
  have e_1552_1 := arithEq_of_rows h (row := 1552) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_1
  simp only [← c95115, ← c95116, k2, ← c95117] at e_1552_1
  have hr := e_1552_1
  simp only [e_1552_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15086)) := by
  have c95118 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95119 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95120 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95121 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95122 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95123 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_2 := arithEq_of_rows h (row := 1552) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_2
  simp only [← c95118, ← c95119, ← c95120] at e_1552_2
  have e_1552_3 := arithEq_of_rows h (row := 1552) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_3
  simp only [← c95121, ← c95122, k2, ← c95123] at e_1552_3
  have hr := e_1552_3
  simp only [e_1552_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15087)) := by
  have c95124 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95125 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95126 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95127 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95128 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95129 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_4 := arithEq_of_rows h (row := 1552) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_4
  simp only [← c95124, ← c95125, ← c95126] at e_1552_4
  have e_1552_5 := arithEq_of_rows h (row := 1552) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_5
  simp only [← c95127, ← c95128, k2, ← c95129] at e_1552_5
  have hr := e_1552_5
  simp only [e_1552_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15088)) := by
  have c95130 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95131 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95132 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95133 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95134 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95135 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_6 := arithEq_of_rows h (row := 1552) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_6
  simp only [← c95130, ← c95131, ← c95132] at e_1552_6
  have e_1552_7 := arithEq_of_rows h (row := 1552) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_7
  simp only [← c95133, ← c95134, k2, ← c95135] at e_1552_7
  have hr := e_1552_7
  simp only [e_1552_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15089)) := by
  have c95136 := (publicBatchWrapper89_copies2973 a h).1
  have c95137 := (publicBatchWrapper89_copies2973 a h).2.1
  have c95138 := (publicBatchWrapper89_copies2973 a h).2.2.1
  have c95139 := (publicBatchWrapper89_copies2973 a h).2.2.2.1
  have c95140 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.1
  have c95141 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_8 := arithEq_of_rows h (row := 1552) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_8
  simp only [← c95136, ← c95137, ← c95138] at e_1552_8
  have e_1552_9 := arithEq_of_rows h (row := 1552) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_9
  simp only [← c95139, ← c95140, k2, ← c95141] at e_1552_9
  have hr := e_1552_9
  simp only [e_1552_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15090)) := by
  have c95142 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.1
  have c95143 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.1
  have c95144 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.1
  have c95145 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.1
  have c95146 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95147 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_10 := arithEq_of_rows h (row := 1552) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_10
  simp only [← c95142, ← c95143, ← c95144] at e_1552_10
  have e_1552_11 := arithEq_of_rows h (row := 1552) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_11
  simp only [← c95145, ← c95146, k2, ← c95147] at e_1552_11
  have hr := e_1552_11
  simp only [e_1552_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15091)) := by
  have c95148 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95149 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95150 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95151 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95152 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95153 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_12 := arithEq_of_rows h (row := 1552) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_12
  simp only [← c95148, ← c95149, ← c95150] at e_1552_12
  have e_1552_13 := arithEq_of_rows h (row := 1552) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_13
  simp only [← c95151, ← c95152, k2, ← c95153] at e_1552_13
  have hr := e_1552_13
  simp only [e_1552_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15092)) := by
  have c95154 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95155 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95156 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95157 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95158 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95159 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_14 := arithEq_of_rows h (row := 1552) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_14
  simp only [← c95154, ← c95155, ← c95156] at e_1552_14
  have e_1552_15 := arithEq_of_rows h (row := 1552) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_15
  simp only [← c95157, ← c95158, k2, ← c95159] at e_1552_15
  have hr := e_1552_15
  simp only [e_1552_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15093)) := by
  have c95160 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95161 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95162 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95163 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95164 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95165 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_16 := arithEq_of_rows h (row := 1552) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_16
  simp only [← c95160, ← c95161, ← c95162] at e_1552_16
  have e_1552_17 := arithEq_of_rows h (row := 1552) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_17
  simp only [← c95163, ← c95164, k2, ← c95165] at e_1552_17
  have hr := e_1552_17
  simp only [e_1552_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15094)) := by
  have c95166 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95167 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95168 := (publicBatchWrapper89_copies2974 a h).1
  have c95169 := (publicBatchWrapper89_copies2974 a h).2.1
  have c95170 := (publicBatchWrapper89_copies2974 a h).2.2.1
  have c95171 := (publicBatchWrapper89_copies2974 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_18 := arithEq_of_rows h (row := 1552) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_18
  simp only [← c95166, ← c95167, ← c95168] at e_1552_18
  have e_1552_19 := arithEq_of_rows h (row := 1552) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_19
  simp only [← c95169, ← c95170, k2, ← c95171] at e_1552_19
  have hr := e_1552_19
  simp only [e_1552_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15095)) := by
  have c95172 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.1
  have c95173 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.1
  have c95174 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.1
  have c95175 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.1
  have c95176 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.1
  have c95177 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_0 := arithEq_of_rows h (row := 1553) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_0
  simp only [← c95172, ← c95173, ← c95174] at e_1553_0
  have e_1553_1 := arithEq_of_rows h (row := 1553) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_1
  simp only [← c95175, ← c95176, k2, ← c95177] at e_1553_1
  have hr := e_1553_1
  simp only [e_1553_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15096)) := by
  have c95178 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95179 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95180 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95181 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95182 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95183 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_2 := arithEq_of_rows h (row := 1553) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_2
  simp only [← c95178, ← c95179, ← c95180] at e_1553_2
  have e_1553_3 := arithEq_of_rows h (row := 1553) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_3
  simp only [← c95181, ← c95182, k2, ← c95183] at e_1553_3
  have hr := e_1553_3
  simp only [e_1553_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15097)) := by
  have c95184 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95185 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95186 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95187 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95188 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95189 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_4 := arithEq_of_rows h (row := 1553) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_4
  simp only [← c95184, ← c95185, ← c95186] at e_1553_4
  have e_1553_5 := arithEq_of_rows h (row := 1553) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_5
  simp only [← c95187, ← c95188, k2, ← c95189] at e_1553_5
  have hr := e_1553_5
  simp only [e_1553_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15098)) := by
  have c95190 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95191 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95192 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95193 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95194 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95195 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_6 := arithEq_of_rows h (row := 1553) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_6
  simp only [← c95190, ← c95191, ← c95192] at e_1553_6
  have e_1553_7 := arithEq_of_rows h (row := 1553) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_7
  simp only [← c95193, ← c95194, k2, ← c95195] at e_1553_7
  have hr := e_1553_7
  simp only [e_1553_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15099)) := by
  have c95196 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95197 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95198 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95199 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95200 := (publicBatchWrapper89_copies2975 a h).1
  have c95201 := (publicBatchWrapper89_copies2975 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_8 := arithEq_of_rows h (row := 1553) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_8
  simp only [← c95196, ← c95197, ← c95198] at e_1553_8
  have e_1553_9 := arithEq_of_rows h (row := 1553) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_9
  simp only [← c95199, ← c95200, k2, ← c95201] at e_1553_9
  have hr := e_1553_9
  simp only [e_1553_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15100)) := by
  have c95202 := (publicBatchWrapper89_copies2975 a h).2.2.1
  have c95203 := (publicBatchWrapper89_copies2975 a h).2.2.2.1
  have c95204 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.1
  have c95205 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.1
  have c95206 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.1
  have c95207 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_10 := arithEq_of_rows h (row := 1553) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_10
  simp only [← c95202, ← c95203, ← c95204] at e_1553_10
  have e_1553_11 := arithEq_of_rows h (row := 1553) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_11
  simp only [← c95205, ← c95206, k2, ← c95207] at e_1553_11
  have hr := e_1553_11
  simp only [e_1553_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15101)) := by
  have c95208 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.1
  have c95209 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.1
  have c95210 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95211 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95212 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95213 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_12 := arithEq_of_rows h (row := 1553) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_12
  simp only [← c95208, ← c95209, ← c95210] at e_1553_12
  have e_1553_13 := arithEq_of_rows h (row := 1553) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_13
  simp only [← c95211, ← c95212, k2, ← c95213] at e_1553_13
  have hr := e_1553_13
  simp only [e_1553_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15102)) := by
  have c95214 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95215 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95216 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95217 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95218 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95219 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_14 := arithEq_of_rows h (row := 1553) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_14
  simp only [← c95214, ← c95215, ← c95216] at e_1553_14
  have e_1553_15 := arithEq_of_rows h (row := 1553) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_15
  simp only [← c95217, ← c95218, k2, ← c95219] at e_1553_15
  have hr := e_1553_15
  simp only [e_1553_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15103)) := by
  have c95220 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95221 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95222 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95223 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95224 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95225 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_16 := arithEq_of_rows h (row := 1553) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_16
  simp only [← c95220, ← c95221, ← c95222] at e_1553_16
  have e_1553_17 := arithEq_of_rows h (row := 1553) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_17
  simp only [← c95223, ← c95224, k2, ← c95225] at e_1553_17
  have hr := e_1553_17
  simp only [e_1553_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15104)) := by
  have c95226 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95227 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95228 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95229 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95230 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95231 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_18 := arithEq_of_rows h (row := 1553) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_18
  simp only [← c95226, ← c95227, ← c95228] at e_1553_18
  have e_1553_19 := arithEq_of_rows h (row := 1553) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_19
  simp only [← c95229, ← c95230, k2, ← c95231] at e_1553_19
  have hr := e_1553_19
  simp only [e_1553_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15105)) := by
  have c95232 := (publicBatchWrapper89_copies2976 a h).1
  have c95233 := (publicBatchWrapper89_copies2976 a h).2.1
  have c95234 := (publicBatchWrapper89_copies2976 a h).2.2.1
  have c95235 := (publicBatchWrapper89_copies2976 a h).2.2.2.1
  have c95236 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.1
  have c95237 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_0 := arithEq_of_rows h (row := 1554) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_0
  simp only [← c95232, ← c95233, ← c95234] at e_1554_0
  have e_1554_1 := arithEq_of_rows h (row := 1554) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_1
  simp only [← c95235, ← c95236, k2, ← c95237] at e_1554_1
  have hr := e_1554_1
  simp only [e_1554_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15106)) := by
  have c95238 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.1
  have c95239 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.1
  have c95240 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.1
  have c95241 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.1
  have c95242 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95243 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_2 := arithEq_of_rows h (row := 1554) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_2
  simp only [← c95238, ← c95239, ← c95240] at e_1554_2
  have e_1554_3 := arithEq_of_rows h (row := 1554) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_3
  simp only [← c95241, ← c95242, k2, ← c95243] at e_1554_3
  have hr := e_1554_3
  simp only [e_1554_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15107)) := by
  have c95244 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95245 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95246 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95247 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95248 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95249 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_4 := arithEq_of_rows h (row := 1554) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_4
  simp only [← c95244, ← c95245, ← c95246] at e_1554_4
  have e_1554_5 := arithEq_of_rows h (row := 1554) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_5
  simp only [← c95247, ← c95248, k2, ← c95249] at e_1554_5
  have hr := e_1554_5
  simp only [e_1554_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15108)) := by
  have c95250 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95251 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95252 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95253 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95254 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95255 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_6 := arithEq_of_rows h (row := 1554) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_6
  simp only [← c95250, ← c95251, ← c95252] at e_1554_6
  have e_1554_7 := arithEq_of_rows h (row := 1554) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_7
  simp only [← c95253, ← c95254, k2, ← c95255] at e_1554_7
  have hr := e_1554_7
  simp only [e_1554_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15109)) := by
  have c95256 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95257 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95258 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95259 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95260 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95261 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_8 := arithEq_of_rows h (row := 1554) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_8
  simp only [← c95256, ← c95257, ← c95258] at e_1554_8
  have e_1554_9 := arithEq_of_rows h (row := 1554) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_9
  simp only [← c95259, ← c95260, k2, ← c95261] at e_1554_9
  have hr := e_1554_9
  simp only [e_1554_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15110)) := by
  have c95262 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95263 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95264 := (publicBatchWrapper89_copies2977 a h).1
  have c95265 := (publicBatchWrapper89_copies2977 a h).2.1
  have c95266 := (publicBatchWrapper89_copies2977 a h).2.2.1
  have c95267 := (publicBatchWrapper89_copies2977 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_10 := arithEq_of_rows h (row := 1554) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_10
  simp only [← c95262, ← c95263, ← c95264] at e_1554_10
  have e_1554_11 := arithEq_of_rows h (row := 1554) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_11
  simp only [← c95265, ← c95266, k2, ← c95267] at e_1554_11
  have hr := e_1554_11
  simp only [e_1554_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15111)) := by
  have c95268 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.1
  have c95269 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.1
  have c95270 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.1
  have c95271 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.1
  have c95272 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.1
  have c95273 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_12 := arithEq_of_rows h (row := 1554) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_12
  simp only [← c95268, ← c95269, ← c95270] at e_1554_12
  have e_1554_13 := arithEq_of_rows h (row := 1554) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_13
  simp only [← c95271, ← c95272, k2, ← c95273] at e_1554_13
  have hr := e_1554_13
  simp only [e_1554_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15112)) := by
  have c95274 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95275 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95276 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95277 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95278 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95279 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_14 := arithEq_of_rows h (row := 1554) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_14
  simp only [← c95274, ← c95275, ← c95276] at e_1554_14
  have e_1554_15 := arithEq_of_rows h (row := 1554) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_15
  simp only [← c95277, ← c95278, k2, ← c95279] at e_1554_15
  have hr := e_1554_15
  simp only [e_1554_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15113)) := by
  have c95280 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95281 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95282 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95283 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95284 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95285 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_16 := arithEq_of_rows h (row := 1554) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_16
  simp only [← c95280, ← c95281, ← c95282] at e_1554_16
  have e_1554_17 := arithEq_of_rows h (row := 1554) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_17
  simp only [← c95283, ← c95284, k2, ← c95285] at e_1554_17
  have hr := e_1554_17
  simp only [e_1554_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15114)) := by
  have c95286 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95287 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95288 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95289 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95290 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95291 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_18 := arithEq_of_rows h (row := 1554) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_18
  simp only [← c95286, ← c95287, ← c95288] at e_1554_18
  have e_1554_19 := arithEq_of_rows h (row := 1554) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_19
  simp only [← c95289, ← c95290, k2, ← c95291] at e_1554_19
  have hr := e_1554_19
  simp only [e_1554_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15115)) := by
  have c95292 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95293 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95294 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95295 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95296 := (publicBatchWrapper89_copies2978 a h).1
  have c95297 := (publicBatchWrapper89_copies2978 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_0 := arithEq_of_rows h (row := 1555) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_0
  simp only [← c95292, ← c95293, ← c95294] at e_1555_0
  have e_1555_1 := arithEq_of_rows h (row := 1555) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_1
  simp only [← c95295, ← c95296, k2, ← c95297] at e_1555_1
  have hr := e_1555_1
  simp only [e_1555_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15116)) := by
  have c95298 := (publicBatchWrapper89_copies2978 a h).2.2.1
  have c95299 := (publicBatchWrapper89_copies2978 a h).2.2.2.1
  have c95300 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.1
  have c95301 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.1
  have c95302 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.1
  have c95303 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_2 := arithEq_of_rows h (row := 1555) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_2
  simp only [← c95298, ← c95299, ← c95300] at e_1555_2
  have e_1555_3 := arithEq_of_rows h (row := 1555) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_3
  simp only [← c95301, ← c95302, k2, ← c95303] at e_1555_3
  have hr := e_1555_3
  simp only [e_1555_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15117)) := by
  have c95304 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.1
  have c95305 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.1
  have c95306 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95307 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95308 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95309 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_4 := arithEq_of_rows h (row := 1555) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_4
  simp only [← c95304, ← c95305, ← c95306] at e_1555_4
  have e_1555_5 := arithEq_of_rows h (row := 1555) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_5
  simp only [← c95307, ← c95308, k2, ← c95309] at e_1555_5
  have hr := e_1555_5
  simp only [e_1555_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15118)) := by
  have c95310 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95311 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95312 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95313 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95314 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95315 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_6 := arithEq_of_rows h (row := 1555) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_6
  simp only [← c95310, ← c95311, ← c95312] at e_1555_6
  have e_1555_7 := arithEq_of_rows h (row := 1555) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_7
  simp only [← c95313, ← c95314, k2, ← c95315] at e_1555_7
  have hr := e_1555_7
  simp only [e_1555_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15119)) := by
  have c95316 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95317 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95318 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95319 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95320 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95321 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_8 := arithEq_of_rows h (row := 1555) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_8
  simp only [← c95316, ← c95317, ← c95318] at e_1555_8
  have e_1555_9 := arithEq_of_rows h (row := 1555) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_9
  simp only [← c95319, ← c95320, k2, ← c95321] at e_1555_9
  have hr := e_1555_9
  simp only [e_1555_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15120)) := by
  have c95322 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95323 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95324 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95325 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95326 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95327 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_10 := arithEq_of_rows h (row := 1555) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_10
  simp only [← c95322, ← c95323, ← c95324] at e_1555_10
  have e_1555_11 := arithEq_of_rows h (row := 1555) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_11
  simp only [← c95325, ← c95326, k2, ← c95327] at e_1555_11
  have hr := e_1555_11
  simp only [e_1555_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15121)) := by
  have c95328 := (publicBatchWrapper89_copies2979 a h).1
  have c95329 := (publicBatchWrapper89_copies2979 a h).2.1
  have c95330 := (publicBatchWrapper89_copies2979 a h).2.2.1
  have c95331 := (publicBatchWrapper89_copies2979 a h).2.2.2.1
  have c95332 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.1
  have c95333 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_12 := arithEq_of_rows h (row := 1555) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_12
  simp only [← c95328, ← c95329, ← c95330] at e_1555_12
  have e_1555_13 := arithEq_of_rows h (row := 1555) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_13
  simp only [← c95331, ← c95332, k2, ← c95333] at e_1555_13
  have hr := e_1555_13
  simp only [e_1555_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15122)) := by
  have c95334 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.1
  have c95335 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.1
  have c95336 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.1
  have c95337 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.1
  have c95338 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95339 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_14 := arithEq_of_rows h (row := 1555) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_14
  simp only [← c95334, ← c95335, ← c95336] at e_1555_14
  have e_1555_15 := arithEq_of_rows h (row := 1555) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_15
  simp only [← c95337, ← c95338, k2, ← c95339] at e_1555_15
  have hr := e_1555_15
  simp only [e_1555_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15123)) := by
  have c95340 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95341 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95342 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95343 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95344 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95345 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_16 := arithEq_of_rows h (row := 1555) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_16
  simp only [← c95340, ← c95341, ← c95342] at e_1555_16
  have e_1555_17 := arithEq_of_rows h (row := 1555) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_17
  simp only [← c95343, ← c95344, k2, ← c95345] at e_1555_17
  have hr := e_1555_17
  simp only [e_1555_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15124)) := by
  have c95346 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95347 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95348 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95349 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95350 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95351 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_18 := arithEq_of_rows h (row := 1555) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_18
  simp only [← c95346, ← c95347, ← c95348] at e_1555_18
  have e_1555_19 := arithEq_of_rows h (row := 1555) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_19
  simp only [← c95349, ← c95350, k2, ← c95351] at e_1555_19
  have hr := e_1555_19
  simp only [e_1555_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15125)) := by
  have c95352 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95353 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95354 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95355 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95356 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95357 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_0 := arithEq_of_rows h (row := 1556) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_0
  simp only [← c95352, ← c95353, ← c95354] at e_1556_0
  have e_1556_1 := arithEq_of_rows h (row := 1556) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_1
  simp only [← c95355, ← c95356, k2, ← c95357] at e_1556_1
  have hr := e_1556_1
  simp only [e_1556_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15126)) := by
  have c95358 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95359 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95360 := (publicBatchWrapper89_copies2980 a h).1
  have c95361 := (publicBatchWrapper89_copies2980 a h).2.1
  have c95362 := (publicBatchWrapper89_copies2980 a h).2.2.1
  have c95363 := (publicBatchWrapper89_copies2980 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_2 := arithEq_of_rows h (row := 1556) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_2
  simp only [← c95358, ← c95359, ← c95360] at e_1556_2
  have e_1556_3 := arithEq_of_rows h (row := 1556) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_3
  simp only [← c95361, ← c95362, k2, ← c95363] at e_1556_3
  have hr := e_1556_3
  simp only [e_1556_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15127)) := by
  have c95364 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.1
  have c95365 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.1
  have c95366 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.1
  have c95367 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.1
  have c95368 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.1
  have c95369 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_4 := arithEq_of_rows h (row := 1556) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_4
  simp only [← c95364, ← c95365, ← c95366] at e_1556_4
  have e_1556_5 := arithEq_of_rows h (row := 1556) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_5
  simp only [← c95367, ← c95368, k2, ← c95369] at e_1556_5
  have hr := e_1556_5
  simp only [e_1556_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15128)) := by
  have c95370 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95371 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95372 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95373 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95374 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95375 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_6 := arithEq_of_rows h (row := 1556) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_6
  simp only [← c95370, ← c95371, ← c95372] at e_1556_6
  have e_1556_7 := arithEq_of_rows h (row := 1556) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_7
  simp only [← c95373, ← c95374, k2, ← c95375] at e_1556_7
  have hr := e_1556_7
  simp only [e_1556_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15129)) := by
  have c95376 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95377 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95378 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95379 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95380 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95381 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_8 := arithEq_of_rows h (row := 1556) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_8
  simp only [← c95376, ← c95377, ← c95378] at e_1556_8
  have e_1556_9 := arithEq_of_rows h (row := 1556) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_9
  simp only [← c95379, ← c95380, k2, ← c95381] at e_1556_9
  have hr := e_1556_9
  simp only [e_1556_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15130)) := by
  have c95382 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95383 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95384 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95385 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95386 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95387 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_10 := arithEq_of_rows h (row := 1556) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_10
  simp only [← c95382, ← c95383, ← c95384] at e_1556_10
  have e_1556_11 := arithEq_of_rows h (row := 1556) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_11
  simp only [← c95385, ← c95386, k2, ← c95387] at e_1556_11
  have hr := e_1556_11
  simp only [e_1556_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15131)) := by
  have c95388 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95389 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95390 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95391 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95392 := (publicBatchWrapper89_copies2981 a h).1
  have c95393 := (publicBatchWrapper89_copies2981 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_12 := arithEq_of_rows h (row := 1556) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_12
  simp only [← c95388, ← c95389, ← c95390] at e_1556_12
  have e_1556_13 := arithEq_of_rows h (row := 1556) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_13
  simp only [← c95391, ← c95392, k2, ← c95393] at e_1556_13
  have hr := e_1556_13
  simp only [e_1556_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15132)) := by
  have c95394 := (publicBatchWrapper89_copies2981 a h).2.2.1
  have c95395 := (publicBatchWrapper89_copies2981 a h).2.2.2.1
  have c95396 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.1
  have c95397 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.1
  have c95398 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.1
  have c95399 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_14 := arithEq_of_rows h (row := 1556) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_14
  simp only [← c95394, ← c95395, ← c95396] at e_1556_14
  have e_1556_15 := arithEq_of_rows h (row := 1556) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_15
  simp only [← c95397, ← c95398, k2, ← c95399] at e_1556_15
  have hr := e_1556_15
  simp only [e_1556_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15133)) := by
  have c95400 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.1
  have c95401 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.1
  have c95402 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95403 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95404 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95405 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_16 := arithEq_of_rows h (row := 1556) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_16
  simp only [← c95400, ← c95401, ← c95402] at e_1556_16
  have e_1556_17 := arithEq_of_rows h (row := 1556) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_17
  simp only [← c95403, ← c95404, k2, ← c95405] at e_1556_17
  have hr := e_1556_17
  simp only [e_1556_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15134)) := by
  have c95406 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95407 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95408 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95409 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95410 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95411 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_18 := arithEq_of_rows h (row := 1556) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_18
  simp only [← c95406, ← c95407, ← c95408] at e_1556_18
  have e_1556_19 := arithEq_of_rows h (row := 1556) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_19
  simp only [← c95409, ← c95410, k2, ← c95411] at e_1556_19
  have hr := e_1556_19
  simp only [e_1556_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15135)) := by
  have c95412 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95413 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95414 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95415 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95416 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95417 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_0 := arithEq_of_rows h (row := 1557) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_0
  simp only [← c95412, ← c95413, ← c95414] at e_1557_0
  have e_1557_1 := arithEq_of_rows h (row := 1557) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_1
  simp only [← c95415, ← c95416, k2, ← c95417] at e_1557_1
  have hr := e_1557_1
  simp only [e_1557_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15136)) := by
  have c95418 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95419 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95420 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95421 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95422 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95423 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_2 := arithEq_of_rows h (row := 1557) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_2
  simp only [← c95418, ← c95419, ← c95420] at e_1557_2
  have e_1557_3 := arithEq_of_rows h (row := 1557) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_3
  simp only [← c95421, ← c95422, k2, ← c95423] at e_1557_3
  have hr := e_1557_3
  simp only [e_1557_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15137)) := by
  have c95424 := (publicBatchWrapper89_copies2982 a h).1
  have c95425 := (publicBatchWrapper89_copies2982 a h).2.1
  have c95426 := (publicBatchWrapper89_copies2982 a h).2.2.1
  have c95427 := (publicBatchWrapper89_copies2982 a h).2.2.2.1
  have c95428 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.1
  have c95429 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_4 := arithEq_of_rows h (row := 1557) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_4
  simp only [← c95424, ← c95425, ← c95426] at e_1557_4
  have e_1557_5 := arithEq_of_rows h (row := 1557) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_5
  simp only [← c95427, ← c95428, k2, ← c95429] at e_1557_5
  have hr := e_1557_5
  simp only [e_1557_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15138)) := by
  have c95430 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.1
  have c95431 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.1
  have c95432 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.1
  have c95433 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.1
  have c95434 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95435 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_6 := arithEq_of_rows h (row := 1557) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_6
  simp only [← c95430, ← c95431, ← c95432] at e_1557_6
  have e_1557_7 := arithEq_of_rows h (row := 1557) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_7
  simp only [← c95433, ← c95434, k2, ← c95435] at e_1557_7
  have hr := e_1557_7
  simp only [e_1557_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15139)) := by
  have c95436 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95437 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95438 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95439 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95440 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95441 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_8 := arithEq_of_rows h (row := 1557) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_8
  simp only [← c95436, ← c95437, ← c95438] at e_1557_8
  have e_1557_9 := arithEq_of_rows h (row := 1557) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_9
  simp only [← c95439, ← c95440, k2, ← c95441] at e_1557_9
  have hr := e_1557_9
  simp only [e_1557_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15140)) := by
  have c95442 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95443 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95444 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95445 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95446 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95447 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_10 := arithEq_of_rows h (row := 1557) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_10
  simp only [← c95442, ← c95443, ← c95444] at e_1557_10
  have e_1557_11 := arithEq_of_rows h (row := 1557) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_11
  simp only [← c95445, ← c95446, k2, ← c95447] at e_1557_11
  have hr := e_1557_11
  simp only [e_1557_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15141)) := by
  have c95448 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95449 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95450 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95451 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95452 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95453 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_12 := arithEq_of_rows h (row := 1557) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_12
  simp only [← c95448, ← c95449, ← c95450] at e_1557_12
  have e_1557_13 := arithEq_of_rows h (row := 1557) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_13
  simp only [← c95451, ← c95452, k2, ← c95453] at e_1557_13
  have hr := e_1557_13
  simp only [e_1557_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15142)) := by
  have c95454 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95455 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95456 := (publicBatchWrapper89_copies2983 a h).1
  have c95457 := (publicBatchWrapper89_copies2983 a h).2.1
  have c95458 := (publicBatchWrapper89_copies2983 a h).2.2.1
  have c95459 := (publicBatchWrapper89_copies2983 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_14 := arithEq_of_rows h (row := 1557) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_14
  simp only [← c95454, ← c95455, ← c95456] at e_1557_14
  have e_1557_15 := arithEq_of_rows h (row := 1557) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_15
  simp only [← c95457, ← c95458, k2, ← c95459] at e_1557_15
  have hr := e_1557_15
  simp only [e_1557_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15143)) := by
  have c95460 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.1
  have c95461 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.1
  have c95462 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.1
  have c95463 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.1
  have c95464 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.1
  have c95465 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_16 := arithEq_of_rows h (row := 1557) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_16
  simp only [← c95460, ← c95461, ← c95462] at e_1557_16
  have e_1557_17 := arithEq_of_rows h (row := 1557) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_17
  simp only [← c95463, ← c95464, k2, ← c95465] at e_1557_17
  have hr := e_1557_17
  simp only [e_1557_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15144)) := by
  have c95466 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95467 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95468 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95469 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95470 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95471 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_18 := arithEq_of_rows h (row := 1557) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_18
  simp only [← c95466, ← c95467, ← c95468] at e_1557_18
  have e_1557_19 := arithEq_of_rows h (row := 1557) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_19
  simp only [← c95469, ← c95470, k2, ← c95471] at e_1557_19
  have hr := e_1557_19
  simp only [e_1557_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15145)) := by
  have c95472 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95473 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95474 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95475 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95476 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95477 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_0 := arithEq_of_rows h (row := 1558) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_0
  simp only [← c95472, ← c95473, ← c95474] at e_1558_0
  have e_1558_1 := arithEq_of_rows h (row := 1558) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_1
  simp only [← c95475, ← c95476, k2, ← c95477] at e_1558_1
  have hr := e_1558_1
  simp only [e_1558_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15146)) := by
  have c95478 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95479 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95480 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95481 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95482 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95483 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_2 := arithEq_of_rows h (row := 1558) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_2
  simp only [← c95478, ← c95479, ← c95480] at e_1558_2
  have e_1558_3 := arithEq_of_rows h (row := 1558) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_3
  simp only [← c95481, ← c95482, k2, ← c95483] at e_1558_3
  have hr := e_1558_3
  simp only [e_1558_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15147)) := by
  have c95484 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95485 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95486 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95487 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95488 := (publicBatchWrapper89_copies2984 a h).1
  have c95489 := (publicBatchWrapper89_copies2984 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_4 := arithEq_of_rows h (row := 1558) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_4
  simp only [← c95484, ← c95485, ← c95486] at e_1558_4
  have e_1558_5 := arithEq_of_rows h (row := 1558) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_5
  simp only [← c95487, ← c95488, k2, ← c95489] at e_1558_5
  have hr := e_1558_5
  simp only [e_1558_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15148)) := by
  have c95490 := (publicBatchWrapper89_copies2984 a h).2.2.1
  have c95491 := (publicBatchWrapper89_copies2984 a h).2.2.2.1
  have c95492 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.1
  have c95493 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.1
  have c95494 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.1
  have c95495 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_6 := arithEq_of_rows h (row := 1558) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_6
  simp only [← c95490, ← c95491, ← c95492] at e_1558_6
  have e_1558_7 := arithEq_of_rows h (row := 1558) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_7
  simp only [← c95493, ← c95494, k2, ← c95495] at e_1558_7
  have hr := e_1558_7
  simp only [e_1558_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15149)) := by
  have c95496 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.1
  have c95497 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.1
  have c95498 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95499 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95500 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95501 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_8 := arithEq_of_rows h (row := 1558) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_8
  simp only [← c95496, ← c95497, ← c95498] at e_1558_8
  have e_1558_9 := arithEq_of_rows h (row := 1558) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_9
  simp only [← c95499, ← c95500, k2, ← c95501] at e_1558_9
  have hr := e_1558_9
  simp only [e_1558_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15150)) := by
  have c95502 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95503 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95504 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95505 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95506 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95507 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_10 := arithEq_of_rows h (row := 1558) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_10
  simp only [← c95502, ← c95503, ← c95504] at e_1558_10
  have e_1558_11 := arithEq_of_rows h (row := 1558) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_11
  simp only [← c95505, ← c95506, k2, ← c95507] at e_1558_11
  have hr := e_1558_11
  simp only [e_1558_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15151)) := by
  have c95508 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95509 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95510 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95511 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95512 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95513 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_12 := arithEq_of_rows h (row := 1558) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_12
  simp only [← c95508, ← c95509, ← c95510] at e_1558_12
  have e_1558_13 := arithEq_of_rows h (row := 1558) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_13
  simp only [← c95511, ← c95512, k2, ← c95513] at e_1558_13
  have hr := e_1558_13
  simp only [e_1558_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15152)) := by
  have c95514 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95515 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95516 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95517 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95518 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95519 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_14 := arithEq_of_rows h (row := 1558) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_14
  simp only [← c95514, ← c95515, ← c95516] at e_1558_14
  have e_1558_15 := arithEq_of_rows h (row := 1558) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_15
  simp only [← c95517, ← c95518, k2, ← c95519] at e_1558_15
  have hr := e_1558_15
  simp only [e_1558_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15153)) := by
  have c95520 := (publicBatchWrapper89_copies2985 a h).1
  have c95521 := (publicBatchWrapper89_copies2985 a h).2.1
  have c95522 := (publicBatchWrapper89_copies2985 a h).2.2.1
  have c95523 := (publicBatchWrapper89_copies2985 a h).2.2.2.1
  have c95524 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.1
  have c95525 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_16 := arithEq_of_rows h (row := 1558) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_16
  simp only [← c95520, ← c95521, ← c95522] at e_1558_16
  have e_1558_17 := arithEq_of_rows h (row := 1558) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_17
  simp only [← c95523, ← c95524, k2, ← c95525] at e_1558_17
  have hr := e_1558_17
  simp only [e_1558_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15154)) := by
  have c95526 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.1
  have c95527 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.1
  have c95528 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.1
  have c95529 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.1
  have c95530 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95531 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_18 := arithEq_of_rows h (row := 1558) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_18
  simp only [← c95526, ← c95527, ← c95528] at e_1558_18
  have e_1558_19 := arithEq_of_rows h (row := 1558) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_19
  simp only [← c95529, ← c95530, k2, ← c95531] at e_1558_19
  have hr := e_1558_19
  simp only [e_1558_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15155)) := by
  have c95532 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95533 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95534 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95535 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95536 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95537 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_0 := arithEq_of_rows h (row := 1559) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_0
  simp only [← c95532, ← c95533, ← c95534] at e_1559_0
  have e_1559_1 := arithEq_of_rows h (row := 1559) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_1
  simp only [← c95535, ← c95536, k2, ← c95537] at e_1559_1
  have hr := e_1559_1
  simp only [e_1559_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15156)) := by
  have c95538 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95539 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95540 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95541 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95542 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95543 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_2 := arithEq_of_rows h (row := 1559) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_2
  simp only [← c95538, ← c95539, ← c95540] at e_1559_2
  have e_1559_3 := arithEq_of_rows h (row := 1559) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_3
  simp only [← c95541, ← c95542, k2, ← c95543] at e_1559_3
  have hr := e_1559_3
  simp only [e_1559_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15157)) := by
  have c95544 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95545 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95546 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95547 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95548 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95549 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_4 := arithEq_of_rows h (row := 1559) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_4
  simp only [← c95544, ← c95545, ← c95546] at e_1559_4
  have e_1559_5 := arithEq_of_rows h (row := 1559) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_5
  simp only [← c95547, ← c95548, k2, ← c95549] at e_1559_5
  have hr := e_1559_5
  simp only [e_1559_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15158)) := by
  have c95550 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95551 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95552 := (publicBatchWrapper89_copies2986 a h).1
  have c95553 := (publicBatchWrapper89_copies2986 a h).2.1
  have c95554 := (publicBatchWrapper89_copies2986 a h).2.2.1
  have c95555 := (publicBatchWrapper89_copies2986 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_6 := arithEq_of_rows h (row := 1559) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_6
  simp only [← c95550, ← c95551, ← c95552] at e_1559_6
  have e_1559_7 := arithEq_of_rows h (row := 1559) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_7
  simp only [← c95553, ← c95554, k2, ← c95555] at e_1559_7
  have hr := e_1559_7
  simp only [e_1559_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15159)) := by
  have c95556 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.1
  have c95557 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.1
  have c95558 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.1
  have c95559 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.1
  have c95560 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.1
  have c95561 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_8 := arithEq_of_rows h (row := 1559) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_8
  simp only [← c95556, ← c95557, ← c95558] at e_1559_8
  have e_1559_9 := arithEq_of_rows h (row := 1559) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_9
  simp only [← c95559, ← c95560, k2, ← c95561] at e_1559_9
  have hr := e_1559_9
  simp only [e_1559_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15160)) := by
  have c95562 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95563 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95564 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95565 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95566 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95567 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_10 := arithEq_of_rows h (row := 1559) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_10
  simp only [← c95562, ← c95563, ← c95564] at e_1559_10
  have e_1559_11 := arithEq_of_rows h (row := 1559) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_11
  simp only [← c95565, ← c95566, k2, ← c95567] at e_1559_11
  have hr := e_1559_11
  simp only [e_1559_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15161)) := by
  have c95568 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95569 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95570 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95571 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95572 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95573 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_12 := arithEq_of_rows h (row := 1559) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_12
  simp only [← c95568, ← c95569, ← c95570] at e_1559_12
  have e_1559_13 := arithEq_of_rows h (row := 1559) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_13
  simp only [← c95571, ← c95572, k2, ← c95573] at e_1559_13
  have hr := e_1559_13
  simp only [e_1559_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15162)) := by
  have c95574 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95575 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95576 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95577 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95578 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95579 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_14 := arithEq_of_rows h (row := 1559) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_14
  simp only [← c95574, ← c95575, ← c95576] at e_1559_14
  have e_1559_15 := arithEq_of_rows h (row := 1559) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_15
  simp only [← c95577, ← c95578, k2, ← c95579] at e_1559_15
  have hr := e_1559_15
  simp only [e_1559_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15163)) := by
  have c95580 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95581 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95582 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95583 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95584 := (publicBatchWrapper89_copies2987 a h).1
  have c95585 := (publicBatchWrapper89_copies2987 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_16 := arithEq_of_rows h (row := 1559) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_16
  simp only [← c95580, ← c95581, ← c95582] at e_1559_16
  have e_1559_17 := arithEq_of_rows h (row := 1559) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_17
  simp only [← c95583, ← c95584, k2, ← c95585] at e_1559_17
  have hr := e_1559_17
  simp only [e_1559_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15164)) := by
  have c95586 := (publicBatchWrapper89_copies2987 a h).2.2.1
  have c95587 := (publicBatchWrapper89_copies2987 a h).2.2.2.1
  have c95588 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.1
  have c95589 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.1
  have c95590 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.1
  have c95591 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_18 := arithEq_of_rows h (row := 1559) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_18
  simp only [← c95586, ← c95587, ← c95588] at e_1559_18
  have e_1559_19 := arithEq_of_rows h (row := 1559) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_19
  simp only [← c95589, ← c95590, k2, ← c95591] at e_1559_19
  have hr := e_1559_19
  simp only [e_1559_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15165)) := by
  have c95592 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.1
  have c95593 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.1
  have c95594 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95595 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95596 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95597 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_0 := arithEq_of_rows h (row := 1560) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_0
  simp only [← c95592, ← c95593, ← c95594] at e_1560_0
  have e_1560_1 := arithEq_of_rows h (row := 1560) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_1
  simp only [← c95595, ← c95596, k2, ← c95597] at e_1560_1
  have hr := e_1560_1
  simp only [e_1560_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15166)) := by
  have c95598 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95599 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95600 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95601 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95602 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95603 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_2 := arithEq_of_rows h (row := 1560) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_2
  simp only [← c95598, ← c95599, ← c95600] at e_1560_2
  have e_1560_3 := arithEq_of_rows h (row := 1560) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_3
  simp only [← c95601, ← c95602, k2, ← c95603] at e_1560_3
  have hr := e_1560_3
  simp only [e_1560_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15167)) := by
  have c95604 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95605 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95606 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95607 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95608 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95609 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_4 := arithEq_of_rows h (row := 1560) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_4
  simp only [← c95604, ← c95605, ← c95606] at e_1560_4
  have e_1560_5 := arithEq_of_rows h (row := 1560) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_5
  simp only [← c95607, ← c95608, k2, ← c95609] at e_1560_5
  have hr := e_1560_5
  simp only [e_1560_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15168)) := by
  have c95610 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95611 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95612 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95613 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95614 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95615 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_6 := arithEq_of_rows h (row := 1560) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_6
  simp only [← c95610, ← c95611, ← c95612] at e_1560_6
  have e_1560_7 := arithEq_of_rows h (row := 1560) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_7
  simp only [← c95613, ← c95614, k2, ← c95615] at e_1560_7
  have hr := e_1560_7
  simp only [e_1560_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15169)) := by
  have c95616 := (publicBatchWrapper89_copies2988 a h).1
  have c95617 := (publicBatchWrapper89_copies2988 a h).2.1
  have c95618 := (publicBatchWrapper89_copies2988 a h).2.2.1
  have c95619 := (publicBatchWrapper89_copies2988 a h).2.2.2.1
  have c95620 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.1
  have c95621 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_8 := arithEq_of_rows h (row := 1560) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_8
  simp only [← c95616, ← c95617, ← c95618] at e_1560_8
  have e_1560_9 := arithEq_of_rows h (row := 1560) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_9
  simp only [← c95619, ← c95620, k2, ← c95621] at e_1560_9
  have hr := e_1560_9
  simp only [e_1560_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15210)) := by
  have c95622 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.1
  have c95623 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.1
  have c95624 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.1
  have c95625 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.1
  have c95626 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95627 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_10 := arithEq_of_rows h (row := 1560) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_10
  simp only [← c95622, ← c95623, ← c95624] at e_1560_10
  have e_1560_11 := arithEq_of_rows h (row := 1560) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_11
  simp only [← c95625, ← c95626, k2, ← c95627] at e_1560_11
  have hr := e_1560_11
  simp only [e_1560_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15211)) := by
  have c95628 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95629 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95630 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95631 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95632 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95633 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_12 := arithEq_of_rows h (row := 1560) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_12
  simp only [← c95628, ← c95629, ← c95630] at e_1560_12
  have e_1560_13 := arithEq_of_rows h (row := 1560) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_13
  simp only [← c95631, ← c95632, k2, ← c95633] at e_1560_13
  have hr := e_1560_13
  simp only [e_1560_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15212)) := by
  have c95634 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95635 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95636 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95637 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95638 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95639 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_14 := arithEq_of_rows h (row := 1560) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_14
  simp only [← c95634, ← c95635, ← c95636] at e_1560_14
  have e_1560_15 := arithEq_of_rows h (row := 1560) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_15
  simp only [← c95637, ← c95638, k2, ← c95639] at e_1560_15
  have hr := e_1560_15
  simp only [e_1560_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15213)) := by
  have c95640 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95641 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95642 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95643 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95644 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95645 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_16 := arithEq_of_rows h (row := 1560) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_16
  simp only [← c95640, ← c95641, ← c95642] at e_1560_16
  have e_1560_17 := arithEq_of_rows h (row := 1560) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_17
  simp only [← c95643, ← c95644, k2, ← c95645] at e_1560_17
  have hr := e_1560_17
  simp only [e_1560_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15214)) := by
  have c95646 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95647 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95648 := (publicBatchWrapper89_copies2989 a h).1
  have c95649 := (publicBatchWrapper89_copies2989 a h).2.1
  have c95650 := (publicBatchWrapper89_copies2989 a h).2.2.1
  have c95651 := (publicBatchWrapper89_copies2989 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_18 := arithEq_of_rows h (row := 1560) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_18
  simp only [← c95646, ← c95647, ← c95648] at e_1560_18
  have e_1560_19 := arithEq_of_rows h (row := 1560) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_19
  simp only [← c95649, ← c95650, k2, ← c95651] at e_1560_19
  have hr := e_1560_19
  simp only [e_1560_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15215)) := by
  have c95652 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.1
  have c95653 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.1
  have c95654 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.1
  have c95655 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.1
  have c95656 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.1
  have c95657 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_0 := arithEq_of_rows h (row := 1561) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_0
  simp only [← c95652, ← c95653, ← c95654] at e_1561_0
  have e_1561_1 := arithEq_of_rows h (row := 1561) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_1
  simp only [← c95655, ← c95656, k2, ← c95657] at e_1561_1
  have hr := e_1561_1
  simp only [e_1561_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15216)) := by
  have c95658 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95659 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95660 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95661 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95662 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95663 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_2 := arithEq_of_rows h (row := 1561) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_2
  simp only [← c95658, ← c95659, ← c95660] at e_1561_2
  have e_1561_3 := arithEq_of_rows h (row := 1561) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_3
  simp only [← c95661, ← c95662, k2, ← c95663] at e_1561_3
  have hr := e_1561_3
  simp only [e_1561_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15217)) := by
  have c95664 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95665 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95666 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95667 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95668 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95669 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_4 := arithEq_of_rows h (row := 1561) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_4
  simp only [← c95664, ← c95665, ← c95666] at e_1561_4
  have e_1561_5 := arithEq_of_rows h (row := 1561) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_5
  simp only [← c95667, ← c95668, k2, ← c95669] at e_1561_5
  have hr := e_1561_5
  simp only [e_1561_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15218)) := by
  have c95670 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95671 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95672 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95673 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95674 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95675 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_6 := arithEq_of_rows h (row := 1561) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_6
  simp only [← c95670, ← c95671, ← c95672] at e_1561_6
  have e_1561_7 := arithEq_of_rows h (row := 1561) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_7
  simp only [← c95673, ← c95674, k2, ← c95675] at e_1561_7
  have hr := e_1561_7
  simp only [e_1561_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15219)) := by
  have c95676 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95677 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95678 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95679 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95680 := (publicBatchWrapper89_copies2990 a h).1
  have c95681 := (publicBatchWrapper89_copies2990 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_8 := arithEq_of_rows h (row := 1561) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_8
  simp only [← c95676, ← c95677, ← c95678] at e_1561_8
  have e_1561_9 := arithEq_of_rows h (row := 1561) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_9
  simp only [← c95679, ← c95680, k2, ← c95681] at e_1561_9
  have hr := e_1561_9
  simp only [e_1561_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15220)) := by
  have c95682 := (publicBatchWrapper89_copies2990 a h).2.2.1
  have c95683 := (publicBatchWrapper89_copies2990 a h).2.2.2.1
  have c95684 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.1
  have c95685 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.1
  have c95686 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.1
  have c95687 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_10 := arithEq_of_rows h (row := 1561) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_10
  simp only [← c95682, ← c95683, ← c95684] at e_1561_10
  have e_1561_11 := arithEq_of_rows h (row := 1561) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_11
  simp only [← c95685, ← c95686, k2, ← c95687] at e_1561_11
  have hr := e_1561_11
  simp only [e_1561_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15221)) := by
  have c95688 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.1
  have c95689 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.1
  have c95690 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95691 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95692 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95693 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_12 := arithEq_of_rows h (row := 1561) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_12
  simp only [← c95688, ← c95689, ← c95690] at e_1561_12
  have e_1561_13 := arithEq_of_rows h (row := 1561) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_13
  simp only [← c95691, ← c95692, k2, ← c95693] at e_1561_13
  have hr := e_1561_13
  simp only [e_1561_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15222)) := by
  have c95694 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95695 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95696 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95697 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95698 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95699 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_14 := arithEq_of_rows h (row := 1561) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_14
  simp only [← c95694, ← c95695, ← c95696] at e_1561_14
  have e_1561_15 := arithEq_of_rows h (row := 1561) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_15
  simp only [← c95697, ← c95698, k2, ← c95699] at e_1561_15
  have hr := e_1561_15
  simp only [e_1561_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15223)) := by
  have c95700 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95701 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95702 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95703 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95704 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95705 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_16 := arithEq_of_rows h (row := 1561) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_16
  simp only [← c95700, ← c95701, ← c95702] at e_1561_16
  have e_1561_17 := arithEq_of_rows h (row := 1561) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_17
  simp only [← c95703, ← c95704, k2, ← c95705] at e_1561_17
  have hr := e_1561_17
  simp only [e_1561_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15224)) := by
  have c95706 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95707 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95708 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95709 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95710 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95711 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_18 := arithEq_of_rows h (row := 1561) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_18
  simp only [← c95706, ← c95707, ← c95708] at e_1561_18
  have e_1561_19 := arithEq_of_rows h (row := 1561) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_19
  simp only [← c95709, ← c95710, k2, ← c95711] at e_1561_19
  have hr := e_1561_19
  simp only [e_1561_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15225)) := by
  have c95712 := (publicBatchWrapper89_copies2991 a h).1
  have c95713 := (publicBatchWrapper89_copies2991 a h).2.1
  have c95714 := (publicBatchWrapper89_copies2991 a h).2.2.1
  have c95715 := (publicBatchWrapper89_copies2991 a h).2.2.2.1
  have c95716 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.1
  have c95717 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_0 := arithEq_of_rows h (row := 1562) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_0
  simp only [← c95712, ← c95713, ← c95714] at e_1562_0
  have e_1562_1 := arithEq_of_rows h (row := 1562) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_1
  simp only [← c95715, ← c95716, k2, ← c95717] at e_1562_1
  have hr := e_1562_1
  simp only [e_1562_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15226)) := by
  have c95718 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.1
  have c95719 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.1
  have c95720 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.1
  have c95721 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.1
  have c95722 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95723 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_2 := arithEq_of_rows h (row := 1562) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_2
  simp only [← c95718, ← c95719, ← c95720] at e_1562_2
  have e_1562_3 := arithEq_of_rows h (row := 1562) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_3
  simp only [← c95721, ← c95722, k2, ← c95723] at e_1562_3
  have hr := e_1562_3
  simp only [e_1562_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15227)) := by
  have c95724 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95725 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95726 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95727 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95728 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95729 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_4 := arithEq_of_rows h (row := 1562) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_4
  simp only [← c95724, ← c95725, ← c95726] at e_1562_4
  have e_1562_5 := arithEq_of_rows h (row := 1562) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_5
  simp only [← c95727, ← c95728, k2, ← c95729] at e_1562_5
  have hr := e_1562_5
  simp only [e_1562_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15228)) := by
  have c95730 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95731 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95732 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95733 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95734 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95735 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_6 := arithEq_of_rows h (row := 1562) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_6
  simp only [← c95730, ← c95731, ← c95732] at e_1562_6
  have e_1562_7 := arithEq_of_rows h (row := 1562) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_7
  simp only [← c95733, ← c95734, k2, ← c95735] at e_1562_7
  have hr := e_1562_7
  simp only [e_1562_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15229)) := by
  have c95736 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95737 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95738 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95739 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95740 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95741 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_8 := arithEq_of_rows h (row := 1562) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_8
  simp only [← c95736, ← c95737, ← c95738] at e_1562_8
  have e_1562_9 := arithEq_of_rows h (row := 1562) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_9
  simp only [← c95739, ← c95740, k2, ← c95741] at e_1562_9
  have hr := e_1562_9
  simp only [e_1562_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15230)) := by
  have c95742 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95743 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95744 := (publicBatchWrapper89_copies2992 a h).1
  have c95745 := (publicBatchWrapper89_copies2992 a h).2.1
  have c95746 := (publicBatchWrapper89_copies2992 a h).2.2.1
  have c95747 := (publicBatchWrapper89_copies2992 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_10 := arithEq_of_rows h (row := 1562) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_10
  simp only [← c95742, ← c95743, ← c95744] at e_1562_10
  have e_1562_11 := arithEq_of_rows h (row := 1562) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_11
  simp only [← c95745, ← c95746, k2, ← c95747] at e_1562_11
  have hr := e_1562_11
  simp only [e_1562_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15231)) := by
  have c95748 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.1
  have c95749 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.1
  have c95750 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.1
  have c95751 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.1
  have c95752 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.1
  have c95753 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_12 := arithEq_of_rows h (row := 1562) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_12
  simp only [← c95748, ← c95749, ← c95750] at e_1562_12
  have e_1562_13 := arithEq_of_rows h (row := 1562) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_13
  simp only [← c95751, ← c95752, k2, ← c95753] at e_1562_13
  have hr := e_1562_13
  simp only [e_1562_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15232)) := by
  have c95754 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95755 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95756 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95757 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95758 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95759 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_14 := arithEq_of_rows h (row := 1562) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_14
  simp only [← c95754, ← c95755, ← c95756] at e_1562_14
  have e_1562_15 := arithEq_of_rows h (row := 1562) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_15
  simp only [← c95757, ← c95758, k2, ← c95759] at e_1562_15
  have hr := e_1562_15
  simp only [e_1562_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15233)) := by
  have c95760 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95761 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95762 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95763 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95764 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95765 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_16 := arithEq_of_rows h (row := 1562) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_16
  simp only [← c95760, ← c95761, ← c95762] at e_1562_16
  have e_1562_17 := arithEq_of_rows h (row := 1562) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_17
  simp only [← c95763, ← c95764, k2, ← c95765] at e_1562_17
  have hr := e_1562_17
  simp only [e_1562_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15234)) := by
  have c95766 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95767 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95768 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95769 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95770 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95771 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_18 := arithEq_of_rows h (row := 1562) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_18
  simp only [← c95766, ← c95767, ← c95768] at e_1562_18
  have e_1562_19 := arithEq_of_rows h (row := 1562) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_19
  simp only [← c95769, ← c95770, k2, ← c95771] at e_1562_19
  have hr := e_1562_19
  simp only [e_1562_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15235)) := by
  have c95772 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95773 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95774 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95775 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95776 := (publicBatchWrapper89_copies2993 a h).1
  have c95777 := (publicBatchWrapper89_copies2993 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_0 := arithEq_of_rows h (row := 1563) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_0
  simp only [← c95772, ← c95773, ← c95774] at e_1563_0
  have e_1563_1 := arithEq_of_rows h (row := 1563) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_1
  simp only [← c95775, ← c95776, k2, ← c95777] at e_1563_1
  have hr := e_1563_1
  simp only [e_1563_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15236)) := by
  have c95778 := (publicBatchWrapper89_copies2993 a h).2.2.1
  have c95779 := (publicBatchWrapper89_copies2993 a h).2.2.2.1
  have c95780 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.1
  have c95781 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.1
  have c95782 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.1
  have c95783 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_2 := arithEq_of_rows h (row := 1563) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_2
  simp only [← c95778, ← c95779, ← c95780] at e_1563_2
  have e_1563_3 := arithEq_of_rows h (row := 1563) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_3
  simp only [← c95781, ← c95782, k2, ← c95783] at e_1563_3
  have hr := e_1563_3
  simp only [e_1563_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15237)) := by
  have c95784 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.1
  have c95785 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.1
  have c95786 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95787 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95788 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95789 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_4 := arithEq_of_rows h (row := 1563) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_4
  simp only [← c95784, ← c95785, ← c95786] at e_1563_4
  have e_1563_5 := arithEq_of_rows h (row := 1563) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_5
  simp only [← c95787, ← c95788, k2, ← c95789] at e_1563_5
  have hr := e_1563_5
  simp only [e_1563_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15238)) := by
  have c95790 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95791 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95792 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95793 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95794 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95795 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_6 := arithEq_of_rows h (row := 1563) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_6
  simp only [← c95790, ← c95791, ← c95792] at e_1563_6
  have e_1563_7 := arithEq_of_rows h (row := 1563) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_7
  simp only [← c95793, ← c95794, k2, ← c95795] at e_1563_7
  have hr := e_1563_7
  simp only [e_1563_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15239)) := by
  have c95796 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95797 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95798 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95799 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95800 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95801 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_8 := arithEq_of_rows h (row := 1563) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_8
  simp only [← c95796, ← c95797, ← c95798] at e_1563_8
  have e_1563_9 := arithEq_of_rows h (row := 1563) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_9
  simp only [← c95799, ← c95800, k2, ← c95801] at e_1563_9
  have hr := e_1563_9
  simp only [e_1563_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15240)) := by
  have c95802 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95803 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95804 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95805 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95806 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95807 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_10 := arithEq_of_rows h (row := 1563) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_10
  simp only [← c95802, ← c95803, ← c95804] at e_1563_10
  have e_1563_11 := arithEq_of_rows h (row := 1563) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_11
  simp only [← c95805, ← c95806, k2, ← c95807] at e_1563_11
  have hr := e_1563_11
  simp only [e_1563_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15241)) := by
  have c95808 := (publicBatchWrapper89_copies2994 a h).1
  have c95809 := (publicBatchWrapper89_copies2994 a h).2.1
  have c95810 := (publicBatchWrapper89_copies2994 a h).2.2.1
  have c95811 := (publicBatchWrapper89_copies2994 a h).2.2.2.1
  have c95812 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.1
  have c95813 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_12 := arithEq_of_rows h (row := 1563) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_12
  simp only [← c95808, ← c95809, ← c95810] at e_1563_12
  have e_1563_13 := arithEq_of_rows h (row := 1563) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_13
  simp only [← c95811, ← c95812, k2, ← c95813] at e_1563_13
  have hr := e_1563_13
  simp only [e_1563_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15242)) := by
  have c95814 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.1
  have c95815 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.1
  have c95816 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.1
  have c95817 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.1
  have c95818 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95819 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_14 := arithEq_of_rows h (row := 1563) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_14
  simp only [← c95814, ← c95815, ← c95816] at e_1563_14
  have e_1563_15 := arithEq_of_rows h (row := 1563) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_15
  simp only [← c95817, ← c95818, k2, ← c95819] at e_1563_15
  have hr := e_1563_15
  simp only [e_1563_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15243)) := by
  have c95820 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95821 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95822 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95823 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95824 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95825 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_16 := arithEq_of_rows h (row := 1563) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_16
  simp only [← c95820, ← c95821, ← c95822] at e_1563_16
  have e_1563_17 := arithEq_of_rows h (row := 1563) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_17
  simp only [← c95823, ← c95824, k2, ← c95825] at e_1563_17
  have hr := e_1563_17
  simp only [e_1563_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15244)) := by
  have c95826 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95827 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95828 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95829 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95830 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95831 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_18 := arithEq_of_rows h (row := 1563) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_18
  simp only [← c95826, ← c95827, ← c95828] at e_1563_18
  have e_1563_19 := arithEq_of_rows h (row := 1563) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_19
  simp only [← c95829, ← c95830, k2, ← c95831] at e_1563_19
  have hr := e_1563_19
  simp only [e_1563_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15245)) := by
  have c95832 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95833 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95834 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95835 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95836 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95837 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_0 := arithEq_of_rows h (row := 1564) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_0
  simp only [← c95832, ← c95833, ← c95834] at e_1564_0
  have e_1564_1 := arithEq_of_rows h (row := 1564) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_1
  simp only [← c95835, ← c95836, k2, ← c95837] at e_1564_1
  have hr := e_1564_1
  simp only [e_1564_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15246)) := by
  have c95838 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95839 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95840 := (publicBatchWrapper89_copies2995 a h).1
  have c95841 := (publicBatchWrapper89_copies2995 a h).2.1
  have c95842 := (publicBatchWrapper89_copies2995 a h).2.2.1
  have c95843 := (publicBatchWrapper89_copies2995 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_2 := arithEq_of_rows h (row := 1564) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_2
  simp only [← c95838, ← c95839, ← c95840] at e_1564_2
  have e_1564_3 := arithEq_of_rows h (row := 1564) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_3
  simp only [← c95841, ← c95842, k2, ← c95843] at e_1564_3
  have hr := e_1564_3
  simp only [e_1564_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15247)) := by
  have c95844 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.1
  have c95845 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.1
  have c95846 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.1
  have c95847 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.1
  have c95848 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.1
  have c95849 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_4 := arithEq_of_rows h (row := 1564) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_4
  simp only [← c95844, ← c95845, ← c95846] at e_1564_4
  have e_1564_5 := arithEq_of_rows h (row := 1564) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_5
  simp only [← c95847, ← c95848, k2, ← c95849] at e_1564_5
  have hr := e_1564_5
  simp only [e_1564_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15248)) := by
  have c95850 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95851 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95852 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95853 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95854 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95855 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_6 := arithEq_of_rows h (row := 1564) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_6
  simp only [← c95850, ← c95851, ← c95852] at e_1564_6
  have e_1564_7 := arithEq_of_rows h (row := 1564) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_7
  simp only [← c95853, ← c95854, k2, ← c95855] at e_1564_7
  have hr := e_1564_7
  simp only [e_1564_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15249)) := by
  have c95856 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95857 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95858 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95859 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95860 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95861 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_8 := arithEq_of_rows h (row := 1564) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_8
  simp only [← c95856, ← c95857, ← c95858] at e_1564_8
  have e_1564_9 := arithEq_of_rows h (row := 1564) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_9
  simp only [← c95859, ← c95860, k2, ← c95861] at e_1564_9
  have hr := e_1564_9
  simp only [e_1564_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15250)) := by
  have c95862 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95863 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95864 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95865 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95866 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95867 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_10 := arithEq_of_rows h (row := 1564) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_10
  simp only [← c95862, ← c95863, ← c95864] at e_1564_10
  have e_1564_11 := arithEq_of_rows h (row := 1564) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_11
  simp only [← c95865, ← c95866, k2, ← c95867] at e_1564_11
  have hr := e_1564_11
  simp only [e_1564_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15251)) := by
  have c95868 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95869 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95870 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95871 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95872 := (publicBatchWrapper89_copies2996 a h).1
  have c95873 := (publicBatchWrapper89_copies2996 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_12 := arithEq_of_rows h (row := 1564) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_12
  simp only [← c95868, ← c95869, ← c95870] at e_1564_12
  have e_1564_13 := arithEq_of_rows h (row := 1564) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_13
  simp only [← c95871, ← c95872, k2, ← c95873] at e_1564_13
  have hr := e_1564_13
  simp only [e_1564_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15252)) := by
  have c95874 := (publicBatchWrapper89_copies2996 a h).2.2.1
  have c95875 := (publicBatchWrapper89_copies2996 a h).2.2.2.1
  have c95876 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.1
  have c95877 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.1
  have c95878 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.1
  have c95879 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_14 := arithEq_of_rows h (row := 1564) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_14
  simp only [← c95874, ← c95875, ← c95876] at e_1564_14
  have e_1564_15 := arithEq_of_rows h (row := 1564) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_15
  simp only [← c95877, ← c95878, k2, ← c95879] at e_1564_15
  have hr := e_1564_15
  simp only [e_1564_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15253)) := by
  have c95880 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.1
  have c95881 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.1
  have c95882 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95883 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95884 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95885 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_16 := arithEq_of_rows h (row := 1564) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_16
  simp only [← c95880, ← c95881, ← c95882] at e_1564_16
  have e_1564_17 := arithEq_of_rows h (row := 1564) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_17
  simp only [← c95883, ← c95884, k2, ← c95885] at e_1564_17
  have hr := e_1564_17
  simp only [e_1564_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15254)) := by
  have c95886 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95887 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95888 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95889 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95890 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95891 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_18 := arithEq_of_rows h (row := 1564) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_18
  simp only [← c95886, ← c95887, ← c95888] at e_1564_18
  have e_1564_19 := arithEq_of_rows h (row := 1564) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_19
  simp only [← c95889, ← c95890, k2, ← c95891] at e_1564_19
  have hr := e_1564_19
  simp only [e_1564_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15255)) := by
  have c95892 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95893 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95894 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95895 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95896 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95897 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_0 := arithEq_of_rows h (row := 1565) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_0
  simp only [← c95892, ← c95893, ← c95894] at e_1565_0
  have e_1565_1 := arithEq_of_rows h (row := 1565) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_1
  simp only [← c95895, ← c95896, k2, ← c95897] at e_1565_1
  have hr := e_1565_1
  simp only [e_1565_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15256)) := by
  have c95898 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95899 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95900 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95901 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95902 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95903 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_2 := arithEq_of_rows h (row := 1565) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_2
  simp only [← c95898, ← c95899, ← c95900] at e_1565_2
  have e_1565_3 := arithEq_of_rows h (row := 1565) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_3
  simp only [← c95901, ← c95902, k2, ← c95903] at e_1565_3
  have hr := e_1565_3
  simp only [e_1565_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15257)) := by
  have c95904 := (publicBatchWrapper89_copies2997 a h).1
  have c95905 := (publicBatchWrapper89_copies2997 a h).2.1
  have c95906 := (publicBatchWrapper89_copies2997 a h).2.2.1
  have c95907 := (publicBatchWrapper89_copies2997 a h).2.2.2.1
  have c95908 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.1
  have c95909 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_4 := arithEq_of_rows h (row := 1565) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_4
  simp only [← c95904, ← c95905, ← c95906] at e_1565_4
  have e_1565_5 := arithEq_of_rows h (row := 1565) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_5
  simp only [← c95907, ← c95908, k2, ← c95909] at e_1565_5
  have hr := e_1565_5
  simp only [e_1565_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15258)) := by
  have c95910 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.1
  have c95911 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.1
  have c95912 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.1
  have c95913 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.1
  have c95914 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95915 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_6 := arithEq_of_rows h (row := 1565) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_6
  simp only [← c95910, ← c95911, ← c95912] at e_1565_6
  have e_1565_7 := arithEq_of_rows h (row := 1565) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_7
  simp only [← c95913, ← c95914, k2, ← c95915] at e_1565_7
  have hr := e_1565_7
  simp only [e_1565_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15259)) := by
  have c95916 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95917 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95918 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95919 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95920 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95921 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_8 := arithEq_of_rows h (row := 1565) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_8
  simp only [← c95916, ← c95917, ← c95918] at e_1565_8
  have e_1565_9 := arithEq_of_rows h (row := 1565) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_9
  simp only [← c95919, ← c95920, k2, ← c95921] at e_1565_9
  have hr := e_1565_9
  simp only [e_1565_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15260)) := by
  have c95922 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95923 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95924 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95925 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95926 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95927 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_10 := arithEq_of_rows h (row := 1565) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_10
  simp only [← c95922, ← c95923, ← c95924] at e_1565_10
  have e_1565_11 := arithEq_of_rows h (row := 1565) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_11
  simp only [← c95925, ← c95926, k2, ← c95927] at e_1565_11
  have hr := e_1565_11
  simp only [e_1565_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15261)) := by
  have c95928 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95929 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95930 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95931 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95932 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95933 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_12 := arithEq_of_rows h (row := 1565) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_12
  simp only [← c95928, ← c95929, ← c95930] at e_1565_12
  have e_1565_13 := arithEq_of_rows h (row := 1565) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_13
  simp only [← c95931, ← c95932, k2, ← c95933] at e_1565_13
  have hr := e_1565_13
  simp only [e_1565_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15262)) := by
  have c95934 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95935 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95936 := (publicBatchWrapper89_copies2998 a h).1
  have c95937 := (publicBatchWrapper89_copies2998 a h).2.1
  have c95938 := (publicBatchWrapper89_copies2998 a h).2.2.1
  have c95939 := (publicBatchWrapper89_copies2998 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_14 := arithEq_of_rows h (row := 1565) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_14
  simp only [← c95934, ← c95935, ← c95936] at e_1565_14
  have e_1565_15 := arithEq_of_rows h (row := 1565) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_15
  simp only [← c95937, ← c95938, k2, ← c95939] at e_1565_15
  have hr := e_1565_15
  simp only [e_1565_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15263)) := by
  have c95940 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.1
  have c95941 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.1
  have c95942 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.1
  have c95943 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.1
  have c95944 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.1
  have c95945 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_16 := arithEq_of_rows h (row := 1565) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_16
  simp only [← c95940, ← c95941, ← c95942] at e_1565_16
  have e_1565_17 := arithEq_of_rows h (row := 1565) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_17
  simp only [← c95943, ← c95944, k2, ← c95945] at e_1565_17
  have hr := e_1565_17
  simp only [e_1565_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15264)) := by
  have c95946 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95947 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95948 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95949 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95950 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95951 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_18 := arithEq_of_rows h (row := 1565) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_18
  simp only [← c95946, ← c95947, ← c95948] at e_1565_18
  have e_1565_19 := arithEq_of_rows h (row := 1565) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_19
  simp only [← c95949, ← c95950, k2, ← c95951] at e_1565_19
  have hr := e_1565_19
  simp only [e_1565_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15265)) := by
  have c95952 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95953 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95954 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95955 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95956 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95957 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_0 := arithEq_of_rows h (row := 1566) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_0
  simp only [← c95952, ← c95953, ← c95954] at e_1566_0
  have e_1566_1 := arithEq_of_rows h (row := 1566) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_1
  simp only [← c95955, ← c95956, k2, ← c95957] at e_1566_1
  have hr := e_1566_1
  simp only [e_1566_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15266)) := by
  have c95958 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95959 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95960 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95961 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95962 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95963 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_2 := arithEq_of_rows h (row := 1566) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_2
  simp only [← c95958, ← c95959, ← c95960] at e_1566_2
  have e_1566_3 := arithEq_of_rows h (row := 1566) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_3
  simp only [← c95961, ← c95962, k2, ← c95963] at e_1566_3
  have hr := e_1566_3
  simp only [e_1566_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15267)) := by
  have c95964 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95965 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95966 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95967 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95968 := (publicBatchWrapper89_copies2999 a h).1
  have c95969 := (publicBatchWrapper89_copies2999 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_4 := arithEq_of_rows h (row := 1566) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_4
  simp only [← c95964, ← c95965, ← c95966] at e_1566_4
  have e_1566_5 := arithEq_of_rows h (row := 1566) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_5
  simp only [← c95967, ← c95968, k2, ← c95969] at e_1566_5
  have hr := e_1566_5
  simp only [e_1566_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15268)) := by
  have c95970 := (publicBatchWrapper89_copies2999 a h).2.2.1
  have c95971 := (publicBatchWrapper89_copies2999 a h).2.2.2.1
  have c95972 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.1
  have c95973 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.1
  have c95974 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.1
  have c95975 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_6 := arithEq_of_rows h (row := 1566) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_6
  simp only [← c95970, ← c95971, ← c95972] at e_1566_6
  have e_1566_7 := arithEq_of_rows h (row := 1566) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_7
  simp only [← c95973, ← c95974, k2, ← c95975] at e_1566_7
  have hr := e_1566_7
  simp only [e_1566_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15269)) := by
  have c95976 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.1
  have c95977 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.1
  have c95978 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95979 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95980 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95981 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_8 := arithEq_of_rows h (row := 1566) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_8
  simp only [← c95976, ← c95977, ← c95978] at e_1566_8
  have e_1566_9 := arithEq_of_rows h (row := 1566) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_9
  simp only [← c95979, ← c95980, k2, ← c95981] at e_1566_9
  have hr := e_1566_9
  simp only [e_1566_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15270)) := by
  have c95982 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95983 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95984 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95985 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95986 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95987 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_10 := arithEq_of_rows h (row := 1566) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_10
  simp only [← c95982, ← c95983, ← c95984] at e_1566_10
  have e_1566_11 := arithEq_of_rows h (row := 1566) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_11
  simp only [← c95985, ← c95986, k2, ← c95987] at e_1566_11
  have hr := e_1566_11
  simp only [e_1566_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15271)) := by
  have c95988 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95989 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95990 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95991 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95992 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95993 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_12 := arithEq_of_rows h (row := 1566) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_12
  simp only [← c95988, ← c95989, ← c95990] at e_1566_12
  have e_1566_13 := arithEq_of_rows h (row := 1566) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_13
  simp only [← c95991, ← c95992, k2, ← c95993] at e_1566_13
  have hr := e_1566_13
  simp only [e_1566_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15272)) := by
  have c95994 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95995 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95996 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95997 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95998 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95999 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_14 := arithEq_of_rows h (row := 1566) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_14
  simp only [← c95994, ← c95995, ← c95996] at e_1566_14
  have e_1566_15 := arithEq_of_rows h (row := 1566) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_15
  simp only [← c95997, ← c95998, k2, ← c95999] at e_1566_15
  have hr := e_1566_15
  simp only [e_1566_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15273)) := by
  have c96000 := (publicBatchWrapper89_copies3000 a h).1
  have c96001 := (publicBatchWrapper89_copies3000 a h).2.1
  have c96002 := (publicBatchWrapper89_copies3000 a h).2.2.1
  have c96003 := (publicBatchWrapper89_copies3000 a h).2.2.2.1
  have c96004 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.1
  have c96005 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_16 := arithEq_of_rows h (row := 1566) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_16
  simp only [← c96000, ← c96001, ← c96002] at e_1566_16
  have e_1566_17 := arithEq_of_rows h (row := 1566) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_17
  simp only [← c96003, ← c96004, k2, ← c96005] at e_1566_17
  have hr := e_1566_17
  simp only [e_1566_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15274)) := by
  have c96006 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.1
  have c96007 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.1
  have c96008 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.1
  have c96009 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.1
  have c96010 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96011 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_18 := arithEq_of_rows h (row := 1566) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_18
  simp only [← c96006, ← c96007, ← c96008] at e_1566_18
  have e_1566_19 := arithEq_of_rows h (row := 1566) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_19
  simp only [← c96009, ← c96010, k2, ← c96011] at e_1566_19
  have hr := e_1566_19
  simp only [e_1566_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15275)) := by
  have c96012 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96013 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96014 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96015 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96016 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96017 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_0 := arithEq_of_rows h (row := 1567) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_0
  simp only [← c96012, ← c96013, ← c96014] at e_1567_0
  have e_1567_1 := arithEq_of_rows h (row := 1567) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_1
  simp only [← c96015, ← c96016, k2, ← c96017] at e_1567_1
  have hr := e_1567_1
  simp only [e_1567_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15276)) := by
  have c96018 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96019 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96020 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96021 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96022 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96023 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_2 := arithEq_of_rows h (row := 1567) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_2
  simp only [← c96018, ← c96019, ← c96020] at e_1567_2
  have e_1567_3 := arithEq_of_rows h (row := 1567) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_3
  simp only [← c96021, ← c96022, k2, ← c96023] at e_1567_3
  have hr := e_1567_3
  simp only [e_1567_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15277)) := by
  have c96024 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96025 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96026 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96027 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96028 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96029 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_4 := arithEq_of_rows h (row := 1567) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_4
  simp only [← c96024, ← c96025, ← c96026] at e_1567_4
  have e_1567_5 := arithEq_of_rows h (row := 1567) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_5
  simp only [← c96027, ← c96028, k2, ← c96029] at e_1567_5
  have hr := e_1567_5
  simp only [e_1567_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15278)) := by
  have c96030 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96031 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96032 := (publicBatchWrapper89_copies3001 a h).1
  have c96033 := (publicBatchWrapper89_copies3001 a h).2.1
  have c96034 := (publicBatchWrapper89_copies3001 a h).2.2.1
  have c96035 := (publicBatchWrapper89_copies3001 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_6 := arithEq_of_rows h (row := 1567) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_6
  simp only [← c96030, ← c96031, ← c96032] at e_1567_6
  have e_1567_7 := arithEq_of_rows h (row := 1567) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_7
  simp only [← c96033, ← c96034, k2, ← c96035] at e_1567_7
  have hr := e_1567_7
  simp only [e_1567_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15279)) := by
  have c96036 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.1
  have c96037 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.1
  have c96038 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.1
  have c96039 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.1
  have c96040 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.1
  have c96041 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_8 := arithEq_of_rows h (row := 1567) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_8
  simp only [← c96036, ← c96037, ← c96038] at e_1567_8
  have e_1567_9 := arithEq_of_rows h (row := 1567) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_9
  simp only [← c96039, ← c96040, k2, ← c96041] at e_1567_9
  have hr := e_1567_9
  simp only [e_1567_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15280)) := by
  have c96042 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96043 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96044 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96045 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96046 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96047 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_10 := arithEq_of_rows h (row := 1567) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_10
  simp only [← c96042, ← c96043, ← c96044] at e_1567_10
  have e_1567_11 := arithEq_of_rows h (row := 1567) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_11
  simp only [← c96045, ← c96046, k2, ← c96047] at e_1567_11
  have hr := e_1567_11
  simp only [e_1567_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15281)) := by
  have c96048 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96049 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96050 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96051 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96052 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96053 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_12 := arithEq_of_rows h (row := 1567) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_12
  simp only [← c96048, ← c96049, ← c96050] at e_1567_12
  have e_1567_13 := arithEq_of_rows h (row := 1567) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_13
  simp only [← c96051, ← c96052, k2, ← c96053] at e_1567_13
  have hr := e_1567_13
  simp only [e_1567_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15282)) := by
  have c96054 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96055 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96056 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96057 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96058 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96059 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_14 := arithEq_of_rows h (row := 1567) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_14
  simp only [← c96054, ← c96055, ← c96056] at e_1567_14
  have e_1567_15 := arithEq_of_rows h (row := 1567) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_15
  simp only [← c96057, ← c96058, k2, ← c96059] at e_1567_15
  have hr := e_1567_15
  simp only [e_1567_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15283)) := by
  have c96060 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96061 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96062 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96063 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96064 := (publicBatchWrapper89_copies3002 a h).1
  have c96065 := (publicBatchWrapper89_copies3002 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_16 := arithEq_of_rows h (row := 1567) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_16
  simp only [← c96060, ← c96061, ← c96062] at e_1567_16
  have e_1567_17 := arithEq_of_rows h (row := 1567) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_17
  simp only [← c96063, ← c96064, k2, ← c96065] at e_1567_17
  have hr := e_1567_17
  simp only [e_1567_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15284)) := by
  have c96066 := (publicBatchWrapper89_copies3002 a h).2.2.1
  have c96067 := (publicBatchWrapper89_copies3002 a h).2.2.2.1
  have c96068 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.1
  have c96069 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.1
  have c96070 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.1
  have c96071 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_18 := arithEq_of_rows h (row := 1567) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_18
  simp only [← c96066, ← c96067, ← c96068] at e_1567_18
  have e_1567_19 := arithEq_of_rows h (row := 1567) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_19
  simp only [← c96069, ← c96070, k2, ← c96071] at e_1567_19
  have hr := e_1567_19
  simp only [e_1567_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15285)) := by
  have c96072 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.1
  have c96073 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.1
  have c96074 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96075 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96076 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96077 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_0 := arithEq_of_rows h (row := 1568) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_0
  simp only [← c96072, ← c96073, ← c96074] at e_1568_0
  have e_1568_1 := arithEq_of_rows h (row := 1568) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_1
  simp only [← c96075, ← c96076, k2, ← c96077] at e_1568_1
  have hr := e_1568_1
  simp only [e_1568_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15286)) := by
  have c96078 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96079 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96080 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96081 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96082 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96083 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_2 := arithEq_of_rows h (row := 1568) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_2
  simp only [← c96078, ← c96079, ← c96080] at e_1568_2
  have e_1568_3 := arithEq_of_rows h (row := 1568) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_3
  simp only [← c96081, ← c96082, k2, ← c96083] at e_1568_3
  have hr := e_1568_3
  simp only [e_1568_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15287)) := by
  have c96084 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96085 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96086 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96087 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96088 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96089 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_4 := arithEq_of_rows h (row := 1568) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_4
  simp only [← c96084, ← c96085, ← c96086] at e_1568_4
  have e_1568_5 := arithEq_of_rows h (row := 1568) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_5
  simp only [← c96087, ← c96088, k2, ← c96089] at e_1568_5
  have hr := e_1568_5
  simp only [e_1568_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15288)) := by
  have c96090 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96091 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96092 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96093 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96094 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96095 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_6 := arithEq_of_rows h (row := 1568) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_6
  simp only [← c96090, ← c96091, ← c96092] at e_1568_6
  have e_1568_7 := arithEq_of_rows h (row := 1568) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_7
  simp only [← c96093, ← c96094, k2, ← c96095] at e_1568_7
  have hr := e_1568_7
  simp only [e_1568_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15289)) := by
  have c96096 := (publicBatchWrapper89_copies3003 a h).1
  have c96097 := (publicBatchWrapper89_copies3003 a h).2.1
  have c96098 := (publicBatchWrapper89_copies3003 a h).2.2.1
  have c96099 := (publicBatchWrapper89_copies3003 a h).2.2.2.1
  have c96100 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.1
  have c96101 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_8 := arithEq_of_rows h (row := 1568) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_8
  simp only [← c96096, ← c96097, ← c96098] at e_1568_8
  have e_1568_9 := arithEq_of_rows h (row := 1568) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_9
  simp only [← c96099, ← c96100, k2, ← c96101] at e_1568_9
  have hr := e_1568_9
  simp only [e_1568_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15290)) := by
  have c96102 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.1
  have c96103 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.1
  have c96104 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.1
  have c96105 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.1
  have c96106 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96107 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_10 := arithEq_of_rows h (row := 1568) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_10
  simp only [← c96102, ← c96103, ← c96104] at e_1568_10
  have e_1568_11 := arithEq_of_rows h (row := 1568) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_11
  simp only [← c96105, ← c96106, k2, ← c96107] at e_1568_11
  have hr := e_1568_11
  simp only [e_1568_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15291)) := by
  have c96108 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96109 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96110 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96111 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96112 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96113 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_12 := arithEq_of_rows h (row := 1568) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_12
  simp only [← c96108, ← c96109, ← c96110] at e_1568_12
  have e_1568_13 := arithEq_of_rows h (row := 1568) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_13
  simp only [← c96111, ← c96112, k2, ← c96113] at e_1568_13
  have hr := e_1568_13
  simp only [e_1568_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15292)) := by
  have c96114 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96115 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96116 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96117 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96118 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96119 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_14 := arithEq_of_rows h (row := 1568) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_14
  simp only [← c96114, ← c96115, ← c96116] at e_1568_14
  have e_1568_15 := arithEq_of_rows h (row := 1568) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_15
  simp only [← c96117, ← c96118, k2, ← c96119] at e_1568_15
  have hr := e_1568_15
  simp only [e_1568_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15293)) := by
  have c96120 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96121 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96122 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96123 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96124 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96125 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_16 := arithEq_of_rows h (row := 1568) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_16
  simp only [← c96120, ← c96121, ← c96122] at e_1568_16
  have e_1568_17 := arithEq_of_rows h (row := 1568) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_17
  simp only [← c96123, ← c96124, k2, ← c96125] at e_1568_17
  have hr := e_1568_17
  simp only [e_1568_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15294)) := by
  have c96126 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96127 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96128 := (publicBatchWrapper89_copies3004 a h).1
  have c96129 := (publicBatchWrapper89_copies3004 a h).2.1
  have c96130 := (publicBatchWrapper89_copies3004 a h).2.2.1
  have c96131 := (publicBatchWrapper89_copies3004 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_18 := arithEq_of_rows h (row := 1568) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_18
  simp only [← c96126, ← c96127, ← c96128] at e_1568_18
  have e_1568_19 := arithEq_of_rows h (row := 1568) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_19
  simp only [← c96129, ← c96130, k2, ← c96131] at e_1568_19
  have hr := e_1568_19
  simp only [e_1568_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15295)) := by
  have c96132 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.1
  have c96133 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.1
  have c96134 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.1
  have c96135 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.1
  have c96136 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.1
  have c96137 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_0 := arithEq_of_rows h (row := 1569) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_0
  simp only [← c96132, ← c96133, ← c96134] at e_1569_0
  have e_1569_1 := arithEq_of_rows h (row := 1569) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_1
  simp only [← c96135, ← c96136, k2, ← c96137] at e_1569_1
  have hr := e_1569_1
  simp only [e_1569_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15296)) := by
  have c96138 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96139 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96140 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96141 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96142 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96143 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_2 := arithEq_of_rows h (row := 1569) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_2
  simp only [← c96138, ← c96139, ← c96140] at e_1569_2
  have e_1569_3 := arithEq_of_rows h (row := 1569) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_3
  simp only [← c96141, ← c96142, k2, ← c96143] at e_1569_3
  have hr := e_1569_3
  simp only [e_1569_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15297)) := by
  have c96144 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96145 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96146 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96147 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96148 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96149 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_4 := arithEq_of_rows h (row := 1569) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_4
  simp only [← c96144, ← c96145, ← c96146] at e_1569_4
  have e_1569_5 := arithEq_of_rows h (row := 1569) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_5
  simp only [← c96147, ← c96148, k2, ← c96149] at e_1569_5
  have hr := e_1569_5
  simp only [e_1569_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15298)) := by
  have c96150 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96151 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96152 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96153 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96154 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96155 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_6 := arithEq_of_rows h (row := 1569) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_6
  simp only [← c96150, ← c96151, ← c96152] at e_1569_6
  have e_1569_7 := arithEq_of_rows h (row := 1569) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_7
  simp only [← c96153, ← c96154, k2, ← c96155] at e_1569_7
  have hr := e_1569_7
  simp only [e_1569_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15299)) := by
  have c96156 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96157 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96158 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96159 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96160 := (publicBatchWrapper89_copies3005 a h).1
  have c96161 := (publicBatchWrapper89_copies3005 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_8 := arithEq_of_rows h (row := 1569) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_8
  simp only [← c96156, ← c96157, ← c96158] at e_1569_8
  have e_1569_9 := arithEq_of_rows h (row := 1569) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_9
  simp only [← c96159, ← c96160, k2, ← c96161] at e_1569_9
  have hr := e_1569_9
  simp only [e_1569_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15300)) := by
  have c96162 := (publicBatchWrapper89_copies3005 a h).2.2.1
  have c96163 := (publicBatchWrapper89_copies3005 a h).2.2.2.1
  have c96164 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.1
  have c96165 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.1
  have c96166 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.1
  have c96167 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_10 := arithEq_of_rows h (row := 1569) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_10
  simp only [← c96162, ← c96163, ← c96164] at e_1569_10
  have e_1569_11 := arithEq_of_rows h (row := 1569) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_11
  simp only [← c96165, ← c96166, k2, ← c96167] at e_1569_11
  have hr := e_1569_11
  simp only [e_1569_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15301)) := by
  have c96168 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.1
  have c96169 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.1
  have c96170 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96171 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96172 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96173 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_12 := arithEq_of_rows h (row := 1569) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_12
  simp only [← c96168, ← c96169, ← c96170] at e_1569_12
  have e_1569_13 := arithEq_of_rows h (row := 1569) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_13
  simp only [← c96171, ← c96172, k2, ← c96173] at e_1569_13
  have hr := e_1569_13
  simp only [e_1569_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15302)) := by
  have c96174 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96175 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96176 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96177 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96178 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96179 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_14 := arithEq_of_rows h (row := 1569) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_14
  simp only [← c96174, ← c96175, ← c96176] at e_1569_14
  have e_1569_15 := arithEq_of_rows h (row := 1569) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_15
  simp only [← c96177, ← c96178, k2, ← c96179] at e_1569_15
  have hr := e_1569_15
  simp only [e_1569_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15303)) := by
  have c96180 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96181 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96182 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96183 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96184 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96185 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_16 := arithEq_of_rows h (row := 1569) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_16
  simp only [← c96180, ← c96181, ← c96182] at e_1569_16
  have e_1569_17 := arithEq_of_rows h (row := 1569) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_17
  simp only [← c96183, ← c96184, k2, ← c96185] at e_1569_17
  have hr := e_1569_17
  simp only [e_1569_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15304)) := by
  have c96186 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96187 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96188 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96189 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96190 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96191 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_18 := arithEq_of_rows h (row := 1569) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_18
  simp only [← c96186, ← c96187, ← c96188] at e_1569_18
  have e_1569_19 := arithEq_of_rows h (row := 1569) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_19
  simp only [← c96189, ← c96190, k2, ← c96191] at e_1569_19
  have hr := e_1569_19
  simp only [e_1569_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15305)) := by
  have c96192 := (publicBatchWrapper89_copies3006 a h).1
  have c96193 := (publicBatchWrapper89_copies3006 a h).2.1
  have c96194 := (publicBatchWrapper89_copies3006 a h).2.2.1
  have c96195 := (publicBatchWrapper89_copies3006 a h).2.2.2.1
  have c96196 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.1
  have c96197 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_0 := arithEq_of_rows h (row := 1570) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_0
  simp only [← c96192, ← c96193, ← c96194] at e_1570_0
  have e_1570_1 := arithEq_of_rows h (row := 1570) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_1
  simp only [← c96195, ← c96196, k2, ← c96197] at e_1570_1
  have hr := e_1570_1
  simp only [e_1570_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15306)) := by
  have c96198 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.1
  have c96199 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.1
  have c96200 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.1
  have c96201 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.1
  have c96202 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96203 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_2 := arithEq_of_rows h (row := 1570) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_2
  simp only [← c96198, ← c96199, ← c96200] at e_1570_2
  have e_1570_3 := arithEq_of_rows h (row := 1570) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_3
  simp only [← c96201, ← c96202, k2, ← c96203] at e_1570_3
  have hr := e_1570_3
  simp only [e_1570_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15307)) := by
  have c96204 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96205 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96206 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96207 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96208 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96209 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_4 := arithEq_of_rows h (row := 1570) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_4
  simp only [← c96204, ← c96205, ← c96206] at e_1570_4
  have e_1570_5 := arithEq_of_rows h (row := 1570) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_5
  simp only [← c96207, ← c96208, k2, ← c96209] at e_1570_5
  have hr := e_1570_5
  simp only [e_1570_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15308)) := by
  have c96210 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96211 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96212 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96213 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96214 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96215 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_6 := arithEq_of_rows h (row := 1570) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_6
  simp only [← c96210, ← c96211, ← c96212] at e_1570_6
  have e_1570_7 := arithEq_of_rows h (row := 1570) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_7
  simp only [← c96213, ← c96214, k2, ← c96215] at e_1570_7
  have hr := e_1570_7
  simp only [e_1570_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15309)) := by
  have c96216 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96217 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96218 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96219 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96220 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96221 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_8 := arithEq_of_rows h (row := 1570) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_8
  simp only [← c96216, ← c96217, ← c96218] at e_1570_8
  have e_1570_9 := arithEq_of_rows h (row := 1570) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_9
  simp only [← c96219, ← c96220, k2, ← c96221] at e_1570_9
  have hr := e_1570_9
  simp only [e_1570_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15310)) := by
  have c96222 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96223 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96224 := (publicBatchWrapper89_copies3007 a h).1
  have c96225 := (publicBatchWrapper89_copies3007 a h).2.1
  have c96226 := (publicBatchWrapper89_copies3007 a h).2.2.1
  have c96227 := (publicBatchWrapper89_copies3007 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_10 := arithEq_of_rows h (row := 1570) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_10
  simp only [← c96222, ← c96223, ← c96224] at e_1570_10
  have e_1570_11 := arithEq_of_rows h (row := 1570) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_11
  simp only [← c96225, ← c96226, k2, ← c96227] at e_1570_11
  have hr := e_1570_11
  simp only [e_1570_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15311)) := by
  have c96228 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.1
  have c96229 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.1
  have c96230 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.1
  have c96231 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.1
  have c96232 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.1
  have c96233 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_12 := arithEq_of_rows h (row := 1570) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_12
  simp only [← c96228, ← c96229, ← c96230] at e_1570_12
  have e_1570_13 := arithEq_of_rows h (row := 1570) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_13
  simp only [← c96231, ← c96232, k2, ← c96233] at e_1570_13
  have hr := e_1570_13
  simp only [e_1570_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15312)) := by
  have c96234 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96235 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96236 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96237 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96238 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96239 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_14 := arithEq_of_rows h (row := 1570) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_14
  simp only [← c96234, ← c96235, ← c96236] at e_1570_14
  have e_1570_15 := arithEq_of_rows h (row := 1570) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_15
  simp only [← c96237, ← c96238, k2, ← c96239] at e_1570_15
  have hr := e_1570_15
  simp only [e_1570_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15313)) := by
  have c96240 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96241 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96242 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96243 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96244 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96245 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_16 := arithEq_of_rows h (row := 1570) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_16
  simp only [← c96240, ← c96241, ← c96242] at e_1570_16
  have e_1570_17 := arithEq_of_rows h (row := 1570) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_17
  simp only [← c96243, ← c96244, k2, ← c96245] at e_1570_17
  have hr := e_1570_17
  simp only [e_1570_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15314)) := by
  have c96246 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96247 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96248 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96249 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96250 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96251 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_18 := arithEq_of_rows h (row := 1570) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_18
  simp only [← c96246, ← c96247, ← c96248] at e_1570_18
  have e_1570_19 := arithEq_of_rows h (row := 1570) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_19
  simp only [← c96249, ← c96250, k2, ← c96251] at e_1570_19
  have hr := e_1570_19
  simp only [e_1570_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15315)) := by
  have c96252 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96253 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96254 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96255 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96256 := (publicBatchWrapper89_copies3008 a h).1
  have c96257 := (publicBatchWrapper89_copies3008 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_0 := arithEq_of_rows h (row := 1571) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_0
  simp only [← c96252, ← c96253, ← c96254] at e_1571_0
  have e_1571_1 := arithEq_of_rows h (row := 1571) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_1
  simp only [← c96255, ← c96256, k2, ← c96257] at e_1571_1
  have hr := e_1571_1
  simp only [e_1571_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15316)) := by
  have c96258 := (publicBatchWrapper89_copies3008 a h).2.2.1
  have c96259 := (publicBatchWrapper89_copies3008 a h).2.2.2.1
  have c96260 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.1
  have c96261 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.1
  have c96262 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.1
  have c96263 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_2 := arithEq_of_rows h (row := 1571) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_2
  simp only [← c96258, ← c96259, ← c96260] at e_1571_2
  have e_1571_3 := arithEq_of_rows h (row := 1571) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_3
  simp only [← c96261, ← c96262, k2, ← c96263] at e_1571_3
  have hr := e_1571_3
  simp only [e_1571_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15317)) := by
  have c96264 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.1
  have c96265 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.1
  have c96266 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96267 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96268 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96269 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_4 := arithEq_of_rows h (row := 1571) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_4
  simp only [← c96264, ← c96265, ← c96266] at e_1571_4
  have e_1571_5 := arithEq_of_rows h (row := 1571) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_5
  simp only [← c96267, ← c96268, k2, ← c96269] at e_1571_5
  have hr := e_1571_5
  simp only [e_1571_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15318)) := by
  have c96270 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96271 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96272 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96273 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96274 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96275 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_6 := arithEq_of_rows h (row := 1571) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_6
  simp only [← c96270, ← c96271, ← c96272] at e_1571_6
  have e_1571_7 := arithEq_of_rows h (row := 1571) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_7
  simp only [← c96273, ← c96274, k2, ← c96275] at e_1571_7
  have hr := e_1571_7
  simp only [e_1571_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15319)) := by
  have c96276 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96277 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96278 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96279 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96280 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96281 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_8 := arithEq_of_rows h (row := 1571) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_8
  simp only [← c96276, ← c96277, ← c96278] at e_1571_8
  have e_1571_9 := arithEq_of_rows h (row := 1571) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_9
  simp only [← c96279, ← c96280, k2, ← c96281] at e_1571_9
  have hr := e_1571_9
  simp only [e_1571_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15320)) := by
  have c96282 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96283 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96284 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96285 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96286 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96287 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_10 := arithEq_of_rows h (row := 1571) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_10
  simp only [← c96282, ← c96283, ← c96284] at e_1571_10
  have e_1571_11 := arithEq_of_rows h (row := 1571) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_11
  simp only [← c96285, ← c96286, k2, ← c96287] at e_1571_11
  have hr := e_1571_11
  simp only [e_1571_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15321)) := by
  have c96288 := (publicBatchWrapper89_copies3009 a h).1
  have c96289 := (publicBatchWrapper89_copies3009 a h).2.1
  have c96290 := (publicBatchWrapper89_copies3009 a h).2.2.1
  have c96291 := (publicBatchWrapper89_copies3009 a h).2.2.2.1
  have c96292 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.1
  have c96293 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_12 := arithEq_of_rows h (row := 1571) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_12
  simp only [← c96288, ← c96289, ← c96290] at e_1571_12
  have e_1571_13 := arithEq_of_rows h (row := 1571) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_13
  simp only [← c96291, ← c96292, k2, ← c96293] at e_1571_13
  have hr := e_1571_13
  simp only [e_1571_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15322)) := by
  have c96294 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.1
  have c96295 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.1
  have c96296 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.1
  have c96297 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.1
  have c96298 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96299 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_14 := arithEq_of_rows h (row := 1571) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_14
  simp only [← c96294, ← c96295, ← c96296] at e_1571_14
  have e_1571_15 := arithEq_of_rows h (row := 1571) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_15
  simp only [← c96297, ← c96298, k2, ← c96299] at e_1571_15
  have hr := e_1571_15
  simp only [e_1571_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15323)) := by
  have c96300 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96301 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96302 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96303 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96304 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96305 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_16 := arithEq_of_rows h (row := 1571) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_16
  simp only [← c96300, ← c96301, ← c96302] at e_1571_16
  have e_1571_17 := arithEq_of_rows h (row := 1571) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_17
  simp only [← c96303, ← c96304, k2, ← c96305] at e_1571_17
  have hr := e_1571_17
  simp only [e_1571_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15324)) := by
  have c96306 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96307 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96308 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96309 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96310 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96311 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_18 := arithEq_of_rows h (row := 1571) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_18
  simp only [← c96306, ← c96307, ← c96308] at e_1571_18
  have e_1571_19 := arithEq_of_rows h (row := 1571) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_19
  simp only [← c96309, ← c96310, k2, ← c96311] at e_1571_19
  have hr := e_1571_19
  simp only [e_1571_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15325)) := by
  have c96312 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96313 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96314 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96315 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96316 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96317 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_0 := arithEq_of_rows h (row := 1572) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_0
  simp only [← c96312, ← c96313, ← c96314] at e_1572_0
  have e_1572_1 := arithEq_of_rows h (row := 1572) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_1
  simp only [← c96315, ← c96316, k2, ← c96317] at e_1572_1
  have hr := e_1572_1
  simp only [e_1572_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15326)) := by
  have c96318 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96319 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96320 := (publicBatchWrapper89_copies3010 a h).1
  have c96321 := (publicBatchWrapper89_copies3010 a h).2.1
  have c96322 := (publicBatchWrapper89_copies3010 a h).2.2.1
  have c96323 := (publicBatchWrapper89_copies3010 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_2 := arithEq_of_rows h (row := 1572) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_2
  simp only [← c96318, ← c96319, ← c96320] at e_1572_2
  have e_1572_3 := arithEq_of_rows h (row := 1572) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_3
  simp only [← c96321, ← c96322, k2, ← c96323] at e_1572_3
  have hr := e_1572_3
  simp only [e_1572_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15327)) := by
  have c96324 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.1
  have c96325 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.1
  have c96326 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.1
  have c96327 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.1
  have c96328 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.1
  have c96329 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_4 := arithEq_of_rows h (row := 1572) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_4
  simp only [← c96324, ← c96325, ← c96326] at e_1572_4
  have e_1572_5 := arithEq_of_rows h (row := 1572) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_5
  simp only [← c96327, ← c96328, k2, ← c96329] at e_1572_5
  have hr := e_1572_5
  simp only [e_1572_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15328)) := by
  have c96330 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96331 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96332 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96333 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96334 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96335 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_6 := arithEq_of_rows h (row := 1572) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_6
  simp only [← c96330, ← c96331, ← c96332] at e_1572_6
  have e_1572_7 := arithEq_of_rows h (row := 1572) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_7
  simp only [← c96333, ← c96334, k2, ← c96335] at e_1572_7
  have hr := e_1572_7
  simp only [e_1572_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15329)) := by
  have c96336 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96337 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96338 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96339 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96340 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96341 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_8 := arithEq_of_rows h (row := 1572) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_8
  simp only [← c96336, ← c96337, ← c96338] at e_1572_8
  have e_1572_9 := arithEq_of_rows h (row := 1572) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_9
  simp only [← c96339, ← c96340, k2, ← c96341] at e_1572_9
  have hr := e_1572_9
  simp only [e_1572_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15330)) := by
  have c96342 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96343 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96344 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96345 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96346 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96347 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_10 := arithEq_of_rows h (row := 1572) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_10
  simp only [← c96342, ← c96343, ← c96344] at e_1572_10
  have e_1572_11 := arithEq_of_rows h (row := 1572) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_11
  simp only [← c96345, ← c96346, k2, ← c96347] at e_1572_11
  have hr := e_1572_11
  simp only [e_1572_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15331)) := by
  have c96348 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96349 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96350 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96351 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96352 := (publicBatchWrapper89_copies3011 a h).1
  have c96353 := (publicBatchWrapper89_copies3011 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_12 := arithEq_of_rows h (row := 1572) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_12
  simp only [← c96348, ← c96349, ← c96350] at e_1572_12
  have e_1572_13 := arithEq_of_rows h (row := 1572) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_13
  simp only [← c96351, ← c96352, k2, ← c96353] at e_1572_13
  have hr := e_1572_13
  simp only [e_1572_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15332)) := by
  have c96354 := (publicBatchWrapper89_copies3011 a h).2.2.1
  have c96355 := (publicBatchWrapper89_copies3011 a h).2.2.2.1
  have c96356 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.1
  have c96357 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.1
  have c96358 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.1
  have c96359 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_14 := arithEq_of_rows h (row := 1572) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_14
  simp only [← c96354, ← c96355, ← c96356] at e_1572_14
  have e_1572_15 := arithEq_of_rows h (row := 1572) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_15
  simp only [← c96357, ← c96358, k2, ← c96359] at e_1572_15
  have hr := e_1572_15
  simp only [e_1572_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15333)) := by
  have c96360 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.1
  have c96361 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.1
  have c96362 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96363 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96364 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96365 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_16 := arithEq_of_rows h (row := 1572) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_16
  simp only [← c96360, ← c96361, ← c96362] at e_1572_16
  have e_1572_17 := arithEq_of_rows h (row := 1572) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_17
  simp only [← c96363, ← c96364, k2, ← c96365] at e_1572_17
  have hr := e_1572_17
  simp only [e_1572_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15334)) := by
  have c96366 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96367 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96368 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96369 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96370 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96371 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_18 := arithEq_of_rows h (row := 1572) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_18
  simp only [← c96366, ← c96367, ← c96368] at e_1572_18
  have e_1572_19 := arithEq_of_rows h (row := 1572) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_19
  simp only [← c96369, ← c96370, k2, ← c96371] at e_1572_19
  have hr := e_1572_19
  simp only [e_1572_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15335)) := by
  have c96372 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96373 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96374 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96375 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96376 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96377 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_0 := arithEq_of_rows h (row := 1573) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_0
  simp only [← c96372, ← c96373, ← c96374] at e_1573_0
  have e_1573_1 := arithEq_of_rows h (row := 1573) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_1
  simp only [← c96375, ← c96376, k2, ← c96377] at e_1573_1
  have hr := e_1573_1
  simp only [e_1573_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15336)) := by
  have c96378 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96379 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96380 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96381 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96382 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96383 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_2 := arithEq_of_rows h (row := 1573) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_2
  simp only [← c96378, ← c96379, ← c96380] at e_1573_2
  have e_1573_3 := arithEq_of_rows h (row := 1573) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_3
  simp only [← c96381, ← c96382, k2, ← c96383] at e_1573_3
  have hr := e_1573_3
  simp only [e_1573_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15337)) := by
  have c96384 := (publicBatchWrapper89_copies3012 a h).1
  have c96385 := (publicBatchWrapper89_copies3012 a h).2.1
  have c96386 := (publicBatchWrapper89_copies3012 a h).2.2.1
  have c96387 := (publicBatchWrapper89_copies3012 a h).2.2.2.1
  have c96388 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.1
  have c96389 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_4 := arithEq_of_rows h (row := 1573) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_4
  simp only [← c96384, ← c96385, ← c96386] at e_1573_4
  have e_1573_5 := arithEq_of_rows h (row := 1573) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_5
  simp only [← c96387, ← c96388, k2, ← c96389] at e_1573_5
  have hr := e_1573_5
  simp only [e_1573_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15338)) := by
  have c96390 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.1
  have c96391 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.1
  have c96392 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.1
  have c96393 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.1
  have c96394 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96395 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_6 := arithEq_of_rows h (row := 1573) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_6
  simp only [← c96390, ← c96391, ← c96392] at e_1573_6
  have e_1573_7 := arithEq_of_rows h (row := 1573) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_7
  simp only [← c96393, ← c96394, k2, ← c96395] at e_1573_7
  have hr := e_1573_7
  simp only [e_1573_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15339)) := by
  have c96396 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96397 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96398 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96399 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96400 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96401 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_8 := arithEq_of_rows h (row := 1573) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_8
  simp only [← c96396, ← c96397, ← c96398] at e_1573_8
  have e_1573_9 := arithEq_of_rows h (row := 1573) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_9
  simp only [← c96399, ← c96400, k2, ← c96401] at e_1573_9
  have hr := e_1573_9
  simp only [e_1573_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15340)) := by
  have c96402 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96403 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96404 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96405 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96406 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96407 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_10 := arithEq_of_rows h (row := 1573) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_10
  simp only [← c96402, ← c96403, ← c96404] at e_1573_10
  have e_1573_11 := arithEq_of_rows h (row := 1573) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_11
  simp only [← c96405, ← c96406, k2, ← c96407] at e_1573_11
  have hr := e_1573_11
  simp only [e_1573_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15341)) := by
  have c96408 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96409 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96410 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96411 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96412 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96413 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_12 := arithEq_of_rows h (row := 1573) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_12
  simp only [← c96408, ← c96409, ← c96410] at e_1573_12
  have e_1573_13 := arithEq_of_rows h (row := 1573) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_13
  simp only [← c96411, ← c96412, k2, ← c96413] at e_1573_13
  have hr := e_1573_13
  simp only [e_1573_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15342)) := by
  have c96414 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96415 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96416 := (publicBatchWrapper89_copies3013 a h).1
  have c96417 := (publicBatchWrapper89_copies3013 a h).2.1
  have c96418 := (publicBatchWrapper89_copies3013 a h).2.2.1
  have c96419 := (publicBatchWrapper89_copies3013 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_14 := arithEq_of_rows h (row := 1573) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_14
  simp only [← c96414, ← c96415, ← c96416] at e_1573_14
  have e_1573_15 := arithEq_of_rows h (row := 1573) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_15
  simp only [← c96417, ← c96418, k2, ← c96419] at e_1573_15
  have hr := e_1573_15
  simp only [e_1573_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15343)) := by
  have c96420 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.1
  have c96421 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.1
  have c96422 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.1
  have c96423 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.1
  have c96424 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.1
  have c96425 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_16 := arithEq_of_rows h (row := 1573) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_16
  simp only [← c96420, ← c96421, ← c96422] at e_1573_16
  have e_1573_17 := arithEq_of_rows h (row := 1573) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_17
  simp only [← c96423, ← c96424, k2, ← c96425] at e_1573_17
  have hr := e_1573_17
  simp only [e_1573_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15344)) := by
  have c96426 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96427 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96428 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96429 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96430 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96431 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_18 := arithEq_of_rows h (row := 1573) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_18
  simp only [← c96426, ← c96427, ← c96428] at e_1573_18
  have e_1573_19 := arithEq_of_rows h (row := 1573) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_19
  simp only [← c96429, ← c96430, k2, ← c96431] at e_1573_19
  have hr := e_1573_19
  simp only [e_1573_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15345)) := by
  have c96432 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96433 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96434 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96435 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96436 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96437 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_0 := arithEq_of_rows h (row := 1574) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_0
  simp only [← c96432, ← c96433, ← c96434] at e_1574_0
  have e_1574_1 := arithEq_of_rows h (row := 1574) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_1
  simp only [← c96435, ← c96436, k2, ← c96437] at e_1574_1
  have hr := e_1574_1
  simp only [e_1574_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15346)) := by
  have c96438 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96439 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96440 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96441 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96442 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96443 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_2 := arithEq_of_rows h (row := 1574) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_2
  simp only [← c96438, ← c96439, ← c96440] at e_1574_2
  have e_1574_3 := arithEq_of_rows h (row := 1574) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_3
  simp only [← c96441, ← c96442, k2, ← c96443] at e_1574_3
  have hr := e_1574_3
  simp only [e_1574_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15347)) := by
  have c96444 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96445 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96446 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96447 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96448 := (publicBatchWrapper89_copies3014 a h).1
  have c96449 := (publicBatchWrapper89_copies3014 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_4 := arithEq_of_rows h (row := 1574) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_4
  simp only [← c96444, ← c96445, ← c96446] at e_1574_4
  have e_1574_5 := arithEq_of_rows h (row := 1574) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_5
  simp only [← c96447, ← c96448, k2, ← c96449] at e_1574_5
  have hr := e_1574_5
  simp only [e_1574_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15348)) := by
  have c96450 := (publicBatchWrapper89_copies3014 a h).2.2.1
  have c96451 := (publicBatchWrapper89_copies3014 a h).2.2.2.1
  have c96452 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.1
  have c96453 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.1
  have c96454 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.1
  have c96455 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_6 := arithEq_of_rows h (row := 1574) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_6
  simp only [← c96450, ← c96451, ← c96452] at e_1574_6
  have e_1574_7 := arithEq_of_rows h (row := 1574) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_7
  simp only [← c96453, ← c96454, k2, ← c96455] at e_1574_7
  have hr := e_1574_7
  simp only [e_1574_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15349)) := by
  have c96456 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.1
  have c96457 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.1
  have c96458 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96459 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96460 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96461 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_8 := arithEq_of_rows h (row := 1574) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_8
  simp only [← c96456, ← c96457, ← c96458] at e_1574_8
  have e_1574_9 := arithEq_of_rows h (row := 1574) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_9
  simp only [← c96459, ← c96460, k2, ← c96461] at e_1574_9
  have hr := e_1574_9
  simp only [e_1574_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15350)) := by
  have c96462 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96463 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96464 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96465 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96466 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96467 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_10 := arithEq_of_rows h (row := 1574) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_10
  simp only [← c96462, ← c96463, ← c96464] at e_1574_10
  have e_1574_11 := arithEq_of_rows h (row := 1574) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_11
  simp only [← c96465, ← c96466, k2, ← c96467] at e_1574_11
  have hr := e_1574_11
  simp only [e_1574_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15351)) := by
  have c96468 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96469 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96470 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96471 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96472 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96473 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_12 := arithEq_of_rows h (row := 1574) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_12
  simp only [← c96468, ← c96469, ← c96470] at e_1574_12
  have e_1574_13 := arithEq_of_rows h (row := 1574) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_13
  simp only [← c96471, ← c96472, k2, ← c96473] at e_1574_13
  have hr := e_1574_13
  simp only [e_1574_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15352)) := by
  have c96474 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96475 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96476 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96477 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96478 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96479 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_14 := arithEq_of_rows h (row := 1574) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_14
  simp only [← c96474, ← c96475, ← c96476] at e_1574_14
  have e_1574_15 := arithEq_of_rows h (row := 1574) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_15
  simp only [← c96477, ← c96478, k2, ← c96479] at e_1574_15
  have hr := e_1574_15
  simp only [e_1574_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15353)) := by
  have c96480 := (publicBatchWrapper89_copies3015 a h).1
  have c96481 := (publicBatchWrapper89_copies3015 a h).2.1
  have c96482 := (publicBatchWrapper89_copies3015 a h).2.2.1
  have c96483 := (publicBatchWrapper89_copies3015 a h).2.2.2.1
  have c96484 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.1
  have c96485 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_16 := arithEq_of_rows h (row := 1574) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_16
  simp only [← c96480, ← c96481, ← c96482] at e_1574_16
  have e_1574_17 := arithEq_of_rows h (row := 1574) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_17
  simp only [← c96483, ← c96484, k2, ← c96485] at e_1574_17
  have hr := e_1574_17
  simp only [e_1574_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15354)) := by
  have c96486 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.1
  have c96487 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.1
  have c96488 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.1
  have c96489 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.1
  have c96490 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96491 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_18 := arithEq_of_rows h (row := 1574) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_18
  simp only [← c96486, ← c96487, ← c96488] at e_1574_18
  have e_1574_19 := arithEq_of_rows h (row := 1574) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_19
  simp only [← c96489, ← c96490, k2, ← c96491] at e_1574_19
  have hr := e_1574_19
  simp only [e_1574_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15355)) := by
  have c96492 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96493 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96494 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96495 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96496 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96497 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_0 := arithEq_of_rows h (row := 1575) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_0
  simp only [← c96492, ← c96493, ← c96494] at e_1575_0
  have e_1575_1 := arithEq_of_rows h (row := 1575) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_1
  simp only [← c96495, ← c96496, k2, ← c96497] at e_1575_1
  have hr := e_1575_1
  simp only [e_1575_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15356)) := by
  have c96498 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96499 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96500 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96501 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96502 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96503 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_2 := arithEq_of_rows h (row := 1575) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_2
  simp only [← c96498, ← c96499, ← c96500] at e_1575_2
  have e_1575_3 := arithEq_of_rows h (row := 1575) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_3
  simp only [← c96501, ← c96502, k2, ← c96503] at e_1575_3
  have hr := e_1575_3
  simp only [e_1575_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15357)) := by
  have c96504 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96505 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96506 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96507 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96508 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96509 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_4 := arithEq_of_rows h (row := 1575) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_4
  simp only [← c96504, ← c96505, ← c96506] at e_1575_4
  have e_1575_5 := arithEq_of_rows h (row := 1575) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_5
  simp only [← c96507, ← c96508, k2, ← c96509] at e_1575_5
  have hr := e_1575_5
  simp only [e_1575_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15358)) := by
  have c96510 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96511 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96512 := (publicBatchWrapper89_copies3016 a h).1
  have c96513 := (publicBatchWrapper89_copies3016 a h).2.1
  have c96514 := (publicBatchWrapper89_copies3016 a h).2.2.1
  have c96515 := (publicBatchWrapper89_copies3016 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_6 := arithEq_of_rows h (row := 1575) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_6
  simp only [← c96510, ← c96511, ← c96512] at e_1575_6
  have e_1575_7 := arithEq_of_rows h (row := 1575) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_7
  simp only [← c96513, ← c96514, k2, ← c96515] at e_1575_7
  have hr := e_1575_7
  simp only [e_1575_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15359)) := by
  have c96516 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.1
  have c96517 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.1
  have c96518 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.1
  have c96519 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.1
  have c96520 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.1
  have c96521 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_8 := arithEq_of_rows h (row := 1575) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_8
  simp only [← c96516, ← c96517, ← c96518] at e_1575_8
  have e_1575_9 := arithEq_of_rows h (row := 1575) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_9
  simp only [← c96519, ← c96520, k2, ← c96521] at e_1575_9
  have hr := e_1575_9
  simp only [e_1575_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15360)) := by
  have c96522 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96523 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96524 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96525 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96526 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96527 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_10 := arithEq_of_rows h (row := 1575) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_10
  simp only [← c96522, ← c96523, ← c96524] at e_1575_10
  have e_1575_11 := arithEq_of_rows h (row := 1575) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_11
  simp only [← c96525, ← c96526, k2, ← c96527] at e_1575_11
  have hr := e_1575_11
  simp only [e_1575_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15361)) := by
  have c96528 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96529 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96530 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96531 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96532 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96533 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_12 := arithEq_of_rows h (row := 1575) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_12
  simp only [← c96528, ← c96529, ← c96530] at e_1575_12
  have e_1575_13 := arithEq_of_rows h (row := 1575) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_13
  simp only [← c96531, ← c96532, k2, ← c96533] at e_1575_13
  have hr := e_1575_13
  simp only [e_1575_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15362)) := by
  have c96534 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96535 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96536 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96537 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96538 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96539 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_14 := arithEq_of_rows h (row := 1575) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_14
  simp only [← c96534, ← c96535, ← c96536] at e_1575_14
  have e_1575_15 := arithEq_of_rows h (row := 1575) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_15
  simp only [← c96537, ← c96538, k2, ← c96539] at e_1575_15
  have hr := e_1575_15
  simp only [e_1575_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15363)) := by
  have c96540 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96541 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96542 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96543 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96544 := (publicBatchWrapper89_copies3017 a h).1
  have c96545 := (publicBatchWrapper89_copies3017 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_16 := arithEq_of_rows h (row := 1575) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_16
  simp only [← c96540, ← c96541, ← c96542] at e_1575_16
  have e_1575_17 := arithEq_of_rows h (row := 1575) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_17
  simp only [← c96543, ← c96544, k2, ← c96545] at e_1575_17
  have hr := e_1575_17
  simp only [e_1575_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15364)) := by
  have c96546 := (publicBatchWrapper89_copies3017 a h).2.2.1
  have c96547 := (publicBatchWrapper89_copies3017 a h).2.2.2.1
  have c96548 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.1
  have c96549 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.1
  have c96550 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.1
  have c96551 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_18 := arithEq_of_rows h (row := 1575) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_18
  simp only [← c96546, ← c96547, ← c96548] at e_1575_18
  have e_1575_19 := arithEq_of_rows h (row := 1575) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_19
  simp only [← c96549, ← c96550, k2, ← c96551] at e_1575_19
  have hr := e_1575_19
  simp only [e_1575_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15365)) := by
  have c96552 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.1
  have c96553 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.1
  have c96554 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96555 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96556 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96557 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_0 := arithEq_of_rows h (row := 1576) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_0
  simp only [← c96552, ← c96553, ← c96554] at e_1576_0
  have e_1576_1 := arithEq_of_rows h (row := 1576) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_1
  simp only [← c96555, ← c96556, k2, ← c96557] at e_1576_1
  have hr := e_1576_1
  simp only [e_1576_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15366)) := by
  have c96558 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96559 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96560 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96561 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96562 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96563 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_2 := arithEq_of_rows h (row := 1576) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_2
  simp only [← c96558, ← c96559, ← c96560] at e_1576_2
  have e_1576_3 := arithEq_of_rows h (row := 1576) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_3
  simp only [← c96561, ← c96562, k2, ← c96563] at e_1576_3
  have hr := e_1576_3
  simp only [e_1576_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15367)) := by
  have c96564 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96565 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96566 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96567 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96568 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96569 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_4 := arithEq_of_rows h (row := 1576) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_4
  simp only [← c96564, ← c96565, ← c96566] at e_1576_4
  have e_1576_5 := arithEq_of_rows h (row := 1576) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_5
  simp only [← c96567, ← c96568, k2, ← c96569] at e_1576_5
  have hr := e_1576_5
  simp only [e_1576_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15368)) := by
  have c96570 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96571 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96572 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96573 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96574 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96575 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_6 := arithEq_of_rows h (row := 1576) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_6
  simp only [← c96570, ← c96571, ← c96572] at e_1576_6
  have e_1576_7 := arithEq_of_rows h (row := 1576) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_7
  simp only [← c96573, ← c96574, k2, ← c96575] at e_1576_7
  have hr := e_1576_7
  simp only [e_1576_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15369)) := by
  have c96576 := (publicBatchWrapper89_copies3018 a h).1
  have c96577 := (publicBatchWrapper89_copies3018 a h).2.1
  have c96578 := (publicBatchWrapper89_copies3018 a h).2.2.1
  have c96579 := (publicBatchWrapper89_copies3018 a h).2.2.2.1
  have c96580 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.1
  have c96581 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_8 := arithEq_of_rows h (row := 1576) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_8
  simp only [← c96576, ← c96577, ← c96578] at e_1576_8
  have e_1576_9 := arithEq_of_rows h (row := 1576) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_9
  simp only [← c96579, ← c96580, k2, ← c96581] at e_1576_9
  have hr := e_1576_9
  simp only [e_1576_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15410)) := by
  have c96582 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.1
  have c96583 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.1
  have c96584 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.1
  have c96585 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.1
  have c96586 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96587 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_10 := arithEq_of_rows h (row := 1576) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_10
  simp only [← c96582, ← c96583, ← c96584] at e_1576_10
  have e_1576_11 := arithEq_of_rows h (row := 1576) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_11
  simp only [← c96585, ← c96586, k2, ← c96587] at e_1576_11
  have hr := e_1576_11
  simp only [e_1576_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15411)) := by
  have c96588 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96589 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96590 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96591 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96592 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96593 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_12 := arithEq_of_rows h (row := 1576) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_12
  simp only [← c96588, ← c96589, ← c96590] at e_1576_12
  have e_1576_13 := arithEq_of_rows h (row := 1576) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_13
  simp only [← c96591, ← c96592, k2, ← c96593] at e_1576_13
  have hr := e_1576_13
  simp only [e_1576_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15412)) := by
  have c96594 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96595 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96596 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96597 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96598 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96599 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_14 := arithEq_of_rows h (row := 1576) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_14
  simp only [← c96594, ← c96595, ← c96596] at e_1576_14
  have e_1576_15 := arithEq_of_rows h (row := 1576) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_15
  simp only [← c96597, ← c96598, k2, ← c96599] at e_1576_15
  have hr := e_1576_15
  simp only [e_1576_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15413)) := by
  have c96600 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96601 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96602 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96603 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96604 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96605 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_16 := arithEq_of_rows h (row := 1576) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_16
  simp only [← c96600, ← c96601, ← c96602] at e_1576_16
  have e_1576_17 := arithEq_of_rows h (row := 1576) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_17
  simp only [← c96603, ← c96604, k2, ← c96605] at e_1576_17
  have hr := e_1576_17
  simp only [e_1576_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15414)) := by
  have c96606 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96607 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96608 := (publicBatchWrapper89_copies3019 a h).1
  have c96609 := (publicBatchWrapper89_copies3019 a h).2.1
  have c96610 := (publicBatchWrapper89_copies3019 a h).2.2.1
  have c96611 := (publicBatchWrapper89_copies3019 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_18 := arithEq_of_rows h (row := 1576) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_18
  simp only [← c96606, ← c96607, ← c96608] at e_1576_18
  have e_1576_19 := arithEq_of_rows h (row := 1576) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_19
  simp only [← c96609, ← c96610, k2, ← c96611] at e_1576_19
  have hr := e_1576_19
  simp only [e_1576_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15415)) := by
  have c96612 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.1
  have c96613 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.1
  have c96614 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.1
  have c96615 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.1
  have c96616 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.1
  have c96617 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_0 := arithEq_of_rows h (row := 1577) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_0
  simp only [← c96612, ← c96613, ← c96614] at e_1577_0
  have e_1577_1 := arithEq_of_rows h (row := 1577) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_1
  simp only [← c96615, ← c96616, k2, ← c96617] at e_1577_1
  have hr := e_1577_1
  simp only [e_1577_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15416)) := by
  have c96618 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96619 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96620 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96621 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96622 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96623 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_2 := arithEq_of_rows h (row := 1577) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_2
  simp only [← c96618, ← c96619, ← c96620] at e_1577_2
  have e_1577_3 := arithEq_of_rows h (row := 1577) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_3
  simp only [← c96621, ← c96622, k2, ← c96623] at e_1577_3
  have hr := e_1577_3
  simp only [e_1577_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15417)) := by
  have c96624 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96625 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96626 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96627 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96628 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96629 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_4 := arithEq_of_rows h (row := 1577) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_4
  simp only [← c96624, ← c96625, ← c96626] at e_1577_4
  have e_1577_5 := arithEq_of_rows h (row := 1577) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_5
  simp only [← c96627, ← c96628, k2, ← c96629] at e_1577_5
  have hr := e_1577_5
  simp only [e_1577_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15418)) := by
  have c96630 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96631 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96632 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96633 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96634 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96635 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_6 := arithEq_of_rows h (row := 1577) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_6
  simp only [← c96630, ← c96631, ← c96632] at e_1577_6
  have e_1577_7 := arithEq_of_rows h (row := 1577) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_7
  simp only [← c96633, ← c96634, k2, ← c96635] at e_1577_7
  have hr := e_1577_7
  simp only [e_1577_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15419)) := by
  have c96636 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96637 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96638 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96639 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96640 := (publicBatchWrapper89_copies3020 a h).1
  have c96641 := (publicBatchWrapper89_copies3020 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_8 := arithEq_of_rows h (row := 1577) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_8
  simp only [← c96636, ← c96637, ← c96638] at e_1577_8
  have e_1577_9 := arithEq_of_rows h (row := 1577) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_9
  simp only [← c96639, ← c96640, k2, ← c96641] at e_1577_9
  have hr := e_1577_9
  simp only [e_1577_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15420)) := by
  have c96642 := (publicBatchWrapper89_copies3020 a h).2.2.1
  have c96643 := (publicBatchWrapper89_copies3020 a h).2.2.2.1
  have c96644 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.1
  have c96645 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.1
  have c96646 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.1
  have c96647 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_10 := arithEq_of_rows h (row := 1577) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_10
  simp only [← c96642, ← c96643, ← c96644] at e_1577_10
  have e_1577_11 := arithEq_of_rows h (row := 1577) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_11
  simp only [← c96645, ← c96646, k2, ← c96647] at e_1577_11
  have hr := e_1577_11
  simp only [e_1577_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15421)) := by
  have c96648 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.1
  have c96649 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.1
  have c96650 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96651 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96652 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96653 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_12 := arithEq_of_rows h (row := 1577) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_12
  simp only [← c96648, ← c96649, ← c96650] at e_1577_12
  have e_1577_13 := arithEq_of_rows h (row := 1577) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_13
  simp only [← c96651, ← c96652, k2, ← c96653] at e_1577_13
  have hr := e_1577_13
  simp only [e_1577_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15422)) := by
  have c96654 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96655 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96656 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96657 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96658 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96659 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_14 := arithEq_of_rows h (row := 1577) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_14
  simp only [← c96654, ← c96655, ← c96656] at e_1577_14
  have e_1577_15 := arithEq_of_rows h (row := 1577) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_15
  simp only [← c96657, ← c96658, k2, ← c96659] at e_1577_15
  have hr := e_1577_15
  simp only [e_1577_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15423)) := by
  have c96660 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96661 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96662 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96663 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96664 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96665 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_16 := arithEq_of_rows h (row := 1577) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_16
  simp only [← c96660, ← c96661, ← c96662] at e_1577_16
  have e_1577_17 := arithEq_of_rows h (row := 1577) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_17
  simp only [← c96663, ← c96664, k2, ← c96665] at e_1577_17
  have hr := e_1577_17
  simp only [e_1577_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15424)) := by
  have c96666 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96667 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96668 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96669 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96670 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96671 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_18 := arithEq_of_rows h (row := 1577) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_18
  simp only [← c96666, ← c96667, ← c96668] at e_1577_18
  have e_1577_19 := arithEq_of_rows h (row := 1577) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_19
  simp only [← c96669, ← c96670, k2, ← c96671] at e_1577_19
  have hr := e_1577_19
  simp only [e_1577_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15425)) := by
  have c96672 := (publicBatchWrapper89_copies3021 a h).1
  have c96673 := (publicBatchWrapper89_copies3021 a h).2.1
  have c96674 := (publicBatchWrapper89_copies3021 a h).2.2.1
  have c96675 := (publicBatchWrapper89_copies3021 a h).2.2.2.1
  have c96676 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.1
  have c96677 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_0 := arithEq_of_rows h (row := 1578) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_0
  simp only [← c96672, ← c96673, ← c96674] at e_1578_0
  have e_1578_1 := arithEq_of_rows h (row := 1578) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_1
  simp only [← c96675, ← c96676, k2, ← c96677] at e_1578_1
  have hr := e_1578_1
  simp only [e_1578_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15426)) := by
  have c96678 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.1
  have c96679 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.1
  have c96680 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.1
  have c96681 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.1
  have c96682 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96683 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_2 := arithEq_of_rows h (row := 1578) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_2
  simp only [← c96678, ← c96679, ← c96680] at e_1578_2
  have e_1578_3 := arithEq_of_rows h (row := 1578) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_3
  simp only [← c96681, ← c96682, k2, ← c96683] at e_1578_3
  have hr := e_1578_3
  simp only [e_1578_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15427)) := by
  have c96684 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96685 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96686 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96687 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96688 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96689 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_4 := arithEq_of_rows h (row := 1578) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_4
  simp only [← c96684, ← c96685, ← c96686] at e_1578_4
  have e_1578_5 := arithEq_of_rows h (row := 1578) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_5
  simp only [← c96687, ← c96688, k2, ← c96689] at e_1578_5
  have hr := e_1578_5
  simp only [e_1578_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15428)) := by
  have c96690 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96691 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96692 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96693 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96694 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96695 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_6 := arithEq_of_rows h (row := 1578) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_6
  simp only [← c96690, ← c96691, ← c96692] at e_1578_6
  have e_1578_7 := arithEq_of_rows h (row := 1578) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_7
  simp only [← c96693, ← c96694, k2, ← c96695] at e_1578_7
  have hr := e_1578_7
  simp only [e_1578_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15429)) := by
  have c96696 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96697 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96698 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96699 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96700 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96701 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_8 := arithEq_of_rows h (row := 1578) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_8
  simp only [← c96696, ← c96697, ← c96698] at e_1578_8
  have e_1578_9 := arithEq_of_rows h (row := 1578) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_9
  simp only [← c96699, ← c96700, k2, ← c96701] at e_1578_9
  have hr := e_1578_9
  simp only [e_1578_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15430)) := by
  have c96702 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96703 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96704 := (publicBatchWrapper89_copies3022 a h).1
  have c96705 := (publicBatchWrapper89_copies3022 a h).2.1
  have c96706 := (publicBatchWrapper89_copies3022 a h).2.2.1
  have c96707 := (publicBatchWrapper89_copies3022 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_10 := arithEq_of_rows h (row := 1578) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_10
  simp only [← c96702, ← c96703, ← c96704] at e_1578_10
  have e_1578_11 := arithEq_of_rows h (row := 1578) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_11
  simp only [← c96705, ← c96706, k2, ← c96707] at e_1578_11
  have hr := e_1578_11
  simp only [e_1578_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15431)) := by
  have c96708 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.1
  have c96709 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.1
  have c96710 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.1
  have c96711 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.1
  have c96712 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.1
  have c96713 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_12 := arithEq_of_rows h (row := 1578) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_12
  simp only [← c96708, ← c96709, ← c96710] at e_1578_12
  have e_1578_13 := arithEq_of_rows h (row := 1578) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_13
  simp only [← c96711, ← c96712, k2, ← c96713] at e_1578_13
  have hr := e_1578_13
  simp only [e_1578_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15432)) := by
  have c96714 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96715 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96716 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96717 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96718 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96719 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_14 := arithEq_of_rows h (row := 1578) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_14
  simp only [← c96714, ← c96715, ← c96716] at e_1578_14
  have e_1578_15 := arithEq_of_rows h (row := 1578) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_15
  simp only [← c96717, ← c96718, k2, ← c96719] at e_1578_15
  have hr := e_1578_15
  simp only [e_1578_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15433)) := by
  have c96720 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96721 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96722 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96723 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96724 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96725 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_16 := arithEq_of_rows h (row := 1578) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_16
  simp only [← c96720, ← c96721, ← c96722] at e_1578_16
  have e_1578_17 := arithEq_of_rows h (row := 1578) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_17
  simp only [← c96723, ← c96724, k2, ← c96725] at e_1578_17
  have hr := e_1578_17
  simp only [e_1578_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15434)) := by
  have c96726 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96727 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96728 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96729 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96730 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96731 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_18 := arithEq_of_rows h (row := 1578) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_18
  simp only [← c96726, ← c96727, ← c96728] at e_1578_18
  have e_1578_19 := arithEq_of_rows h (row := 1578) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_19
  simp only [← c96729, ← c96730, k2, ← c96731] at e_1578_19
  have hr := e_1578_19
  simp only [e_1578_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15435)) := by
  have c96732 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96733 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96734 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96735 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96736 := (publicBatchWrapper89_copies3023 a h).1
  have c96737 := (publicBatchWrapper89_copies3023 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_0 := arithEq_of_rows h (row := 1579) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_0
  simp only [← c96732, ← c96733, ← c96734] at e_1579_0
  have e_1579_1 := arithEq_of_rows h (row := 1579) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_1
  simp only [← c96735, ← c96736, k2, ← c96737] at e_1579_1
  have hr := e_1579_1
  simp only [e_1579_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15436)) := by
  have c96738 := (publicBatchWrapper89_copies3023 a h).2.2.1
  have c96739 := (publicBatchWrapper89_copies3023 a h).2.2.2.1
  have c96740 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.1
  have c96741 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.1
  have c96742 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.1
  have c96743 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_2 := arithEq_of_rows h (row := 1579) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_2
  simp only [← c96738, ← c96739, ← c96740] at e_1579_2
  have e_1579_3 := arithEq_of_rows h (row := 1579) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_3
  simp only [← c96741, ← c96742, k2, ← c96743] at e_1579_3
  have hr := e_1579_3
  simp only [e_1579_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15437)) := by
  have c96744 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.1
  have c96745 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.1
  have c96746 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96747 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96748 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96749 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_4 := arithEq_of_rows h (row := 1579) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_4
  simp only [← c96744, ← c96745, ← c96746] at e_1579_4
  have e_1579_5 := arithEq_of_rows h (row := 1579) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_5
  simp only [← c96747, ← c96748, k2, ← c96749] at e_1579_5
  have hr := e_1579_5
  simp only [e_1579_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15438)) := by
  have c96750 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96751 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96752 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96753 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96754 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96755 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_6 := arithEq_of_rows h (row := 1579) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_6
  simp only [← c96750, ← c96751, ← c96752] at e_1579_6
  have e_1579_7 := arithEq_of_rows h (row := 1579) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_7
  simp only [← c96753, ← c96754, k2, ← c96755] at e_1579_7
  have hr := e_1579_7
  simp only [e_1579_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15439)) := by
  have c96756 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96757 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96758 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96759 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96760 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96761 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_8 := arithEq_of_rows h (row := 1579) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_8
  simp only [← c96756, ← c96757, ← c96758] at e_1579_8
  have e_1579_9 := arithEq_of_rows h (row := 1579) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_9
  simp only [← c96759, ← c96760, k2, ← c96761] at e_1579_9
  have hr := e_1579_9
  simp only [e_1579_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15440)) := by
  have c96762 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96763 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96764 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96765 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96766 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96767 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_10 := arithEq_of_rows h (row := 1579) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_10
  simp only [← c96762, ← c96763, ← c96764] at e_1579_10
  have e_1579_11 := arithEq_of_rows h (row := 1579) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_11
  simp only [← c96765, ← c96766, k2, ← c96767] at e_1579_11
  have hr := e_1579_11
  simp only [e_1579_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15441)) := by
  have c96768 := (publicBatchWrapper89_copies3024 a h).1
  have c96769 := (publicBatchWrapper89_copies3024 a h).2.1
  have c96770 := (publicBatchWrapper89_copies3024 a h).2.2.1
  have c96771 := (publicBatchWrapper89_copies3024 a h).2.2.2.1
  have c96772 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.1
  have c96773 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_12 := arithEq_of_rows h (row := 1579) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_12
  simp only [← c96768, ← c96769, ← c96770] at e_1579_12
  have e_1579_13 := arithEq_of_rows h (row := 1579) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_13
  simp only [← c96771, ← c96772, k2, ← c96773] at e_1579_13
  have hr := e_1579_13
  simp only [e_1579_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15442)) := by
  have c96774 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.1
  have c96775 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.1
  have c96776 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.1
  have c96777 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.1
  have c96778 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96779 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_14 := arithEq_of_rows h (row := 1579) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_14
  simp only [← c96774, ← c96775, ← c96776] at e_1579_14
  have e_1579_15 := arithEq_of_rows h (row := 1579) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_15
  simp only [← c96777, ← c96778, k2, ← c96779] at e_1579_15
  have hr := e_1579_15
  simp only [e_1579_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15443)) := by
  have c96780 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96781 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96782 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96783 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96784 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96785 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_16 := arithEq_of_rows h (row := 1579) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_16
  simp only [← c96780, ← c96781, ← c96782] at e_1579_16
  have e_1579_17 := arithEq_of_rows h (row := 1579) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_17
  simp only [← c96783, ← c96784, k2, ← c96785] at e_1579_17
  have hr := e_1579_17
  simp only [e_1579_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15444)) := by
  have c96786 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96787 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96788 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96789 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96790 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96791 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_18 := arithEq_of_rows h (row := 1579) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_18
  simp only [← c96786, ← c96787, ← c96788] at e_1579_18
  have e_1579_19 := arithEq_of_rows h (row := 1579) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_19
  simp only [← c96789, ← c96790, k2, ← c96791] at e_1579_19
  have hr := e_1579_19
  simp only [e_1579_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15445)) := by
  have c96792 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96793 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96794 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96795 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96796 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96797 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_0 := arithEq_of_rows h (row := 1580) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_0
  simp only [← c96792, ← c96793, ← c96794] at e_1580_0
  have e_1580_1 := arithEq_of_rows h (row := 1580) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_1
  simp only [← c96795, ← c96796, k2, ← c96797] at e_1580_1
  have hr := e_1580_1
  simp only [e_1580_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15446)) := by
  have c96798 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96799 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96800 := (publicBatchWrapper89_copies3025 a h).1
  have c96801 := (publicBatchWrapper89_copies3025 a h).2.1
  have c96802 := (publicBatchWrapper89_copies3025 a h).2.2.1
  have c96803 := (publicBatchWrapper89_copies3025 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_2 := arithEq_of_rows h (row := 1580) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_2
  simp only [← c96798, ← c96799, ← c96800] at e_1580_2
  have e_1580_3 := arithEq_of_rows h (row := 1580) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_3
  simp only [← c96801, ← c96802, k2, ← c96803] at e_1580_3
  have hr := e_1580_3
  simp only [e_1580_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15447)) := by
  have c96804 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.1
  have c96805 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.1
  have c96806 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.1
  have c96807 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.1
  have c96808 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.1
  have c96809 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_4 := arithEq_of_rows h (row := 1580) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_4
  simp only [← c96804, ← c96805, ← c96806] at e_1580_4
  have e_1580_5 := arithEq_of_rows h (row := 1580) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_5
  simp only [← c96807, ← c96808, k2, ← c96809] at e_1580_5
  have hr := e_1580_5
  simp only [e_1580_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15448)) := by
  have c96810 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96811 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96812 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96813 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96814 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96815 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_6 := arithEq_of_rows h (row := 1580) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_6
  simp only [← c96810, ← c96811, ← c96812] at e_1580_6
  have e_1580_7 := arithEq_of_rows h (row := 1580) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_7
  simp only [← c96813, ← c96814, k2, ← c96815] at e_1580_7
  have hr := e_1580_7
  simp only [e_1580_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15449)) := by
  have c96816 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96817 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96818 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96819 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96820 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96821 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_8 := arithEq_of_rows h (row := 1580) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_8
  simp only [← c96816, ← c96817, ← c96818] at e_1580_8
  have e_1580_9 := arithEq_of_rows h (row := 1580) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_9
  simp only [← c96819, ← c96820, k2, ← c96821] at e_1580_9
  have hr := e_1580_9
  simp only [e_1580_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15450)) := by
  have c96822 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96823 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96824 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96825 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96826 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96827 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_10 := arithEq_of_rows h (row := 1580) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_10
  simp only [← c96822, ← c96823, ← c96824] at e_1580_10
  have e_1580_11 := arithEq_of_rows h (row := 1580) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_11
  simp only [← c96825, ← c96826, k2, ← c96827] at e_1580_11
  have hr := e_1580_11
  simp only [e_1580_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15451)) := by
  have c96828 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96829 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96830 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96831 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96832 := (publicBatchWrapper89_copies3026 a h).1
  have c96833 := (publicBatchWrapper89_copies3026 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_12 := arithEq_of_rows h (row := 1580) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_12
  simp only [← c96828, ← c96829, ← c96830] at e_1580_12
  have e_1580_13 := arithEq_of_rows h (row := 1580) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_13
  simp only [← c96831, ← c96832, k2, ← c96833] at e_1580_13
  have hr := e_1580_13
  simp only [e_1580_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15452)) := by
  have c96834 := (publicBatchWrapper89_copies3026 a h).2.2.1
  have c96835 := (publicBatchWrapper89_copies3026 a h).2.2.2.1
  have c96836 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.1
  have c96837 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.1
  have c96838 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.1
  have c96839 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_14 := arithEq_of_rows h (row := 1580) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_14
  simp only [← c96834, ← c96835, ← c96836] at e_1580_14
  have e_1580_15 := arithEq_of_rows h (row := 1580) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_15
  simp only [← c96837, ← c96838, k2, ← c96839] at e_1580_15
  have hr := e_1580_15
  simp only [e_1580_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15453)) := by
  have c96840 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.1
  have c96841 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.1
  have c96842 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96843 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96844 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96845 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_16 := arithEq_of_rows h (row := 1580) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_16
  simp only [← c96840, ← c96841, ← c96842] at e_1580_16
  have e_1580_17 := arithEq_of_rows h (row := 1580) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_17
  simp only [← c96843, ← c96844, k2, ← c96845] at e_1580_17
  have hr := e_1580_17
  simp only [e_1580_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15454)) := by
  have c96846 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96847 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96848 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96849 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96850 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96851 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_18 := arithEq_of_rows h (row := 1580) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_18
  simp only [← c96846, ← c96847, ← c96848] at e_1580_18
  have e_1580_19 := arithEq_of_rows h (row := 1580) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_19
  simp only [← c96849, ← c96850, k2, ← c96851] at e_1580_19
  have hr := e_1580_19
  simp only [e_1580_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15455)) := by
  have c96852 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96853 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96854 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96855 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96856 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96857 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_0 := arithEq_of_rows h (row := 1581) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_0
  simp only [← c96852, ← c96853, ← c96854] at e_1581_0
  have e_1581_1 := arithEq_of_rows h (row := 1581) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_1
  simp only [← c96855, ← c96856, k2, ← c96857] at e_1581_1
  have hr := e_1581_1
  simp only [e_1581_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15456)) := by
  have c96858 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96859 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96860 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96861 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96862 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96863 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_2 := arithEq_of_rows h (row := 1581) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_2
  simp only [← c96858, ← c96859, ← c96860] at e_1581_2
  have e_1581_3 := arithEq_of_rows h (row := 1581) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_3
  simp only [← c96861, ← c96862, k2, ← c96863] at e_1581_3
  have hr := e_1581_3
  simp only [e_1581_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15457)) := by
  have c96864 := (publicBatchWrapper89_copies3027 a h).1
  have c96865 := (publicBatchWrapper89_copies3027 a h).2.1
  have c96866 := (publicBatchWrapper89_copies3027 a h).2.2.1
  have c96867 := (publicBatchWrapper89_copies3027 a h).2.2.2.1
  have c96868 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.1
  have c96869 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_4 := arithEq_of_rows h (row := 1581) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_4
  simp only [← c96864, ← c96865, ← c96866] at e_1581_4
  have e_1581_5 := arithEq_of_rows h (row := 1581) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_5
  simp only [← c96867, ← c96868, k2, ← c96869] at e_1581_5
  have hr := e_1581_5
  simp only [e_1581_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15458)) := by
  have c96870 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.1
  have c96871 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.1
  have c96872 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.1
  have c96873 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.1
  have c96874 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96875 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_6 := arithEq_of_rows h (row := 1581) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_6
  simp only [← c96870, ← c96871, ← c96872] at e_1581_6
  have e_1581_7 := arithEq_of_rows h (row := 1581) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_7
  simp only [← c96873, ← c96874, k2, ← c96875] at e_1581_7
  have hr := e_1581_7
  simp only [e_1581_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15459)) := by
  have c96876 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96877 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96878 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96879 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96880 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96881 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_8 := arithEq_of_rows h (row := 1581) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_8
  simp only [← c96876, ← c96877, ← c96878] at e_1581_8
  have e_1581_9 := arithEq_of_rows h (row := 1581) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_9
  simp only [← c96879, ← c96880, k2, ← c96881] at e_1581_9
  have hr := e_1581_9
  simp only [e_1581_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15460)) := by
  have c96882 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96883 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96884 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96885 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96886 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96887 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_10 := arithEq_of_rows h (row := 1581) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_10
  simp only [← c96882, ← c96883, ← c96884] at e_1581_10
  have e_1581_11 := arithEq_of_rows h (row := 1581) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_11
  simp only [← c96885, ← c96886, k2, ← c96887] at e_1581_11
  have hr := e_1581_11
  simp only [e_1581_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15461)) := by
  have c96888 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96889 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96890 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96891 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96892 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96893 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_12 := arithEq_of_rows h (row := 1581) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_12
  simp only [← c96888, ← c96889, ← c96890] at e_1581_12
  have e_1581_13 := arithEq_of_rows h (row := 1581) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_13
  simp only [← c96891, ← c96892, k2, ← c96893] at e_1581_13
  have hr := e_1581_13
  simp only [e_1581_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15462)) := by
  have c96894 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96895 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96896 := (publicBatchWrapper89_copies3028 a h).1
  have c96897 := (publicBatchWrapper89_copies3028 a h).2.1
  have c96898 := (publicBatchWrapper89_copies3028 a h).2.2.1
  have c96899 := (publicBatchWrapper89_copies3028 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_14 := arithEq_of_rows h (row := 1581) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_14
  simp only [← c96894, ← c96895, ← c96896] at e_1581_14
  have e_1581_15 := arithEq_of_rows h (row := 1581) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_15
  simp only [← c96897, ← c96898, k2, ← c96899] at e_1581_15
  have hr := e_1581_15
  simp only [e_1581_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15463)) := by
  have c96900 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.1
  have c96901 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.1
  have c96902 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.1
  have c96903 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.1
  have c96904 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.1
  have c96905 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_16 := arithEq_of_rows h (row := 1581) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_16
  simp only [← c96900, ← c96901, ← c96902] at e_1581_16
  have e_1581_17 := arithEq_of_rows h (row := 1581) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_17
  simp only [← c96903, ← c96904, k2, ← c96905] at e_1581_17
  have hr := e_1581_17
  simp only [e_1581_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15464)) := by
  have c96906 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96907 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96908 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96909 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96910 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96911 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_18 := arithEq_of_rows h (row := 1581) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_18
  simp only [← c96906, ← c96907, ← c96908] at e_1581_18
  have e_1581_19 := arithEq_of_rows h (row := 1581) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_19
  simp only [← c96909, ← c96910, k2, ← c96911] at e_1581_19
  have hr := e_1581_19
  simp only [e_1581_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15465)) := by
  have c96912 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96913 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96914 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96915 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96916 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96917 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_0 := arithEq_of_rows h (row := 1582) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_0
  simp only [← c96912, ← c96913, ← c96914] at e_1582_0
  have e_1582_1 := arithEq_of_rows h (row := 1582) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_1
  simp only [← c96915, ← c96916, k2, ← c96917] at e_1582_1
  have hr := e_1582_1
  simp only [e_1582_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15466)) := by
  have c96918 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96919 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96920 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96921 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96922 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96923 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_2 := arithEq_of_rows h (row := 1582) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_2
  simp only [← c96918, ← c96919, ← c96920] at e_1582_2
  have e_1582_3 := arithEq_of_rows h (row := 1582) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_3
  simp only [← c96921, ← c96922, k2, ← c96923] at e_1582_3
  have hr := e_1582_3
  simp only [e_1582_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15467)) := by
  have c96924 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96925 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96926 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96927 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96928 := (publicBatchWrapper89_copies3029 a h).1
  have c96929 := (publicBatchWrapper89_copies3029 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_4 := arithEq_of_rows h (row := 1582) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_4
  simp only [← c96924, ← c96925, ← c96926] at e_1582_4
  have e_1582_5 := arithEq_of_rows h (row := 1582) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_5
  simp only [← c96927, ← c96928, k2, ← c96929] at e_1582_5
  have hr := e_1582_5
  simp only [e_1582_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15468)) := by
  have c96930 := (publicBatchWrapper89_copies3029 a h).2.2.1
  have c96931 := (publicBatchWrapper89_copies3029 a h).2.2.2.1
  have c96932 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.1
  have c96933 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.1
  have c96934 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.1
  have c96935 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_6 := arithEq_of_rows h (row := 1582) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_6
  simp only [← c96930, ← c96931, ← c96932] at e_1582_6
  have e_1582_7 := arithEq_of_rows h (row := 1582) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_7
  simp only [← c96933, ← c96934, k2, ← c96935] at e_1582_7
  have hr := e_1582_7
  simp only [e_1582_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15469)) := by
  have c96936 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.1
  have c96937 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.1
  have c96938 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96939 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96940 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96941 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_8 := arithEq_of_rows h (row := 1582) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_8
  simp only [← c96936, ← c96937, ← c96938] at e_1582_8
  have e_1582_9 := arithEq_of_rows h (row := 1582) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_9
  simp only [← c96939, ← c96940, k2, ← c96941] at e_1582_9
  have hr := e_1582_9
  simp only [e_1582_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15470)) := by
  have c96942 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96943 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96944 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96945 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96946 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96947 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_10 := arithEq_of_rows h (row := 1582) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_10
  simp only [← c96942, ← c96943, ← c96944] at e_1582_10
  have e_1582_11 := arithEq_of_rows h (row := 1582) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_11
  simp only [← c96945, ← c96946, k2, ← c96947] at e_1582_11
  have hr := e_1582_11
  simp only [e_1582_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15471)) := by
  have c96948 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96949 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96950 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96951 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96952 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96953 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_12 := arithEq_of_rows h (row := 1582) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_12
  simp only [← c96948, ← c96949, ← c96950] at e_1582_12
  have e_1582_13 := arithEq_of_rows h (row := 1582) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_13
  simp only [← c96951, ← c96952, k2, ← c96953] at e_1582_13
  have hr := e_1582_13
  simp only [e_1582_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15472)) := by
  have c96954 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96955 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96956 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96957 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96958 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96959 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_14 := arithEq_of_rows h (row := 1582) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_14
  simp only [← c96954, ← c96955, ← c96956] at e_1582_14
  have e_1582_15 := arithEq_of_rows h (row := 1582) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_15
  simp only [← c96957, ← c96958, k2, ← c96959] at e_1582_15
  have hr := e_1582_15
  simp only [e_1582_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15473)) := by
  have c96960 := (publicBatchWrapper89_copies3030 a h).1
  have c96961 := (publicBatchWrapper89_copies3030 a h).2.1
  have c96962 := (publicBatchWrapper89_copies3030 a h).2.2.1
  have c96963 := (publicBatchWrapper89_copies3030 a h).2.2.2.1
  have c96964 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.1
  have c96965 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_16 := arithEq_of_rows h (row := 1582) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_16
  simp only [← c96960, ← c96961, ← c96962] at e_1582_16
  have e_1582_17 := arithEq_of_rows h (row := 1582) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_17
  simp only [← c96963, ← c96964, k2, ← c96965] at e_1582_17
  have hr := e_1582_17
  simp only [e_1582_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15474)) := by
  have c96966 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.1
  have c96967 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.1
  have c96968 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.1
  have c96969 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.1
  have c96970 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96971 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_18 := arithEq_of_rows h (row := 1582) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_18
  simp only [← c96966, ← c96967, ← c96968] at e_1582_18
  have e_1582_19 := arithEq_of_rows h (row := 1582) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_19
  simp only [← c96969, ← c96970, k2, ← c96971] at e_1582_19
  have hr := e_1582_19
  simp only [e_1582_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15475)) := by
  have c96972 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96973 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96974 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96975 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96976 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96977 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_0 := arithEq_of_rows h (row := 1583) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_0
  simp only [← c96972, ← c96973, ← c96974] at e_1583_0
  have e_1583_1 := arithEq_of_rows h (row := 1583) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_1
  simp only [← c96975, ← c96976, k2, ← c96977] at e_1583_1
  have hr := e_1583_1
  simp only [e_1583_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15476)) := by
  have c96978 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96979 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96980 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96981 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96982 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96983 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_2 := arithEq_of_rows h (row := 1583) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_2
  simp only [← c96978, ← c96979, ← c96980] at e_1583_2
  have e_1583_3 := arithEq_of_rows h (row := 1583) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_3
  simp only [← c96981, ← c96982, k2, ← c96983] at e_1583_3
  have hr := e_1583_3
  simp only [e_1583_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15477)) := by
  have c96984 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96985 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96986 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96987 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96988 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96989 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_4 := arithEq_of_rows h (row := 1583) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_4
  simp only [← c96984, ← c96985, ← c96986] at e_1583_4
  have e_1583_5 := arithEq_of_rows h (row := 1583) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_5
  simp only [← c96987, ← c96988, k2, ← c96989] at e_1583_5
  have hr := e_1583_5
  simp only [e_1583_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15478)) := by
  have c96990 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96991 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96992 := (publicBatchWrapper89_copies3031 a h).1
  have c96993 := (publicBatchWrapper89_copies3031 a h).2.1
  have c96994 := (publicBatchWrapper89_copies3031 a h).2.2.1
  have c96995 := (publicBatchWrapper89_copies3031 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_6 := arithEq_of_rows h (row := 1583) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_6
  simp only [← c96990, ← c96991, ← c96992] at e_1583_6
  have e_1583_7 := arithEq_of_rows h (row := 1583) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_7
  simp only [← c96993, ← c96994, k2, ← c96995] at e_1583_7
  have hr := e_1583_7
  simp only [e_1583_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15479)) := by
  have c96996 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.1
  have c96997 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.1
  have c96998 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.1
  have c96999 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.1
  have c97000 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.1
  have c97001 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_8 := arithEq_of_rows h (row := 1583) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_8
  simp only [← c96996, ← c96997, ← c96998] at e_1583_8
  have e_1583_9 := arithEq_of_rows h (row := 1583) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_9
  simp only [← c96999, ← c97000, k2, ← c97001] at e_1583_9
  have hr := e_1583_9
  simp only [e_1583_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15480)) := by
  have c97002 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97003 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97004 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97005 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97006 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97007 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_10 := arithEq_of_rows h (row := 1583) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_10
  simp only [← c97002, ← c97003, ← c97004] at e_1583_10
  have e_1583_11 := arithEq_of_rows h (row := 1583) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_11
  simp only [← c97005, ← c97006, k2, ← c97007] at e_1583_11
  have hr := e_1583_11
  simp only [e_1583_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15481)) := by
  have c97008 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97009 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97010 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97011 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97012 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97013 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_12 := arithEq_of_rows h (row := 1583) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_12
  simp only [← c97008, ← c97009, ← c97010] at e_1583_12
  have e_1583_13 := arithEq_of_rows h (row := 1583) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_13
  simp only [← c97011, ← c97012, k2, ← c97013] at e_1583_13
  have hr := e_1583_13
  simp only [e_1583_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15482)) := by
  have c97014 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97015 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97016 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97017 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97018 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97019 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_14 := arithEq_of_rows h (row := 1583) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_14
  simp only [← c97014, ← c97015, ← c97016] at e_1583_14
  have e_1583_15 := arithEq_of_rows h (row := 1583) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_15
  simp only [← c97017, ← c97018, k2, ← c97019] at e_1583_15
  have hr := e_1583_15
  simp only [e_1583_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15483)) := by
  have c97020 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97021 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97022 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97023 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97024 := (publicBatchWrapper89_copies3032 a h).1
  have c97025 := (publicBatchWrapper89_copies3032 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_16 := arithEq_of_rows h (row := 1583) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_16
  simp only [← c97020, ← c97021, ← c97022] at e_1583_16
  have e_1583_17 := arithEq_of_rows h (row := 1583) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_17
  simp only [← c97023, ← c97024, k2, ← c97025] at e_1583_17
  have hr := e_1583_17
  simp only [e_1583_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15484)) := by
  have c97026 := (publicBatchWrapper89_copies3032 a h).2.2.1
  have c97027 := (publicBatchWrapper89_copies3032 a h).2.2.2.1
  have c97028 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.1
  have c97029 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.1
  have c97030 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.1
  have c97031 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_18 := arithEq_of_rows h (row := 1583) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_18
  simp only [← c97026, ← c97027, ← c97028] at e_1583_18
  have e_1583_19 := arithEq_of_rows h (row := 1583) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_19
  simp only [← c97029, ← c97030, k2, ← c97031] at e_1583_19
  have hr := e_1583_19
  simp only [e_1583_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15485)) := by
  have c97032 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.1
  have c97033 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.1
  have c97034 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97035 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97036 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97037 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_0 := arithEq_of_rows h (row := 1584) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_0
  simp only [← c97032, ← c97033, ← c97034] at e_1584_0
  have e_1584_1 := arithEq_of_rows h (row := 1584) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_1
  simp only [← c97035, ← c97036, k2, ← c97037] at e_1584_1
  have hr := e_1584_1
  simp only [e_1584_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15486)) := by
  have c97038 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97039 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97040 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97041 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97042 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97043 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_2 := arithEq_of_rows h (row := 1584) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_2
  simp only [← c97038, ← c97039, ← c97040] at e_1584_2
  have e_1584_3 := arithEq_of_rows h (row := 1584) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_3
  simp only [← c97041, ← c97042, k2, ← c97043] at e_1584_3
  have hr := e_1584_3
  simp only [e_1584_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15487)) := by
  have c97044 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97045 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97046 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97047 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97048 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97049 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_4 := arithEq_of_rows h (row := 1584) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_4
  simp only [← c97044, ← c97045, ← c97046] at e_1584_4
  have e_1584_5 := arithEq_of_rows h (row := 1584) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_5
  simp only [← c97047, ← c97048, k2, ← c97049] at e_1584_5
  have hr := e_1584_5
  simp only [e_1584_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15488)) := by
  have c97050 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97051 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97052 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97053 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97054 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97055 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_6 := arithEq_of_rows h (row := 1584) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_6
  simp only [← c97050, ← c97051, ← c97052] at e_1584_6
  have e_1584_7 := arithEq_of_rows h (row := 1584) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_7
  simp only [← c97053, ← c97054, k2, ← c97055] at e_1584_7
  have hr := e_1584_7
  simp only [e_1584_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15489)) := by
  have c97056 := (publicBatchWrapper89_copies3033 a h).1
  have c97057 := (publicBatchWrapper89_copies3033 a h).2.1
  have c97058 := (publicBatchWrapper89_copies3033 a h).2.2.1
  have c97059 := (publicBatchWrapper89_copies3033 a h).2.2.2.1
  have c97060 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.1
  have c97061 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_8 := arithEq_of_rows h (row := 1584) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_8
  simp only [← c97056, ← c97057, ← c97058] at e_1584_8
  have e_1584_9 := arithEq_of_rows h (row := 1584) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_9
  simp only [← c97059, ← c97060, k2, ← c97061] at e_1584_9
  have hr := e_1584_9
  simp only [e_1584_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15490)) := by
  have c97062 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.1
  have c97063 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.1
  have c97064 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.1
  have c97065 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.1
  have c97066 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97067 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_10 := arithEq_of_rows h (row := 1584) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_10
  simp only [← c97062, ← c97063, ← c97064] at e_1584_10
  have e_1584_11 := arithEq_of_rows h (row := 1584) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_11
  simp only [← c97065, ← c97066, k2, ← c97067] at e_1584_11
  have hr := e_1584_11
  simp only [e_1584_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15491)) := by
  have c97068 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97069 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97070 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97071 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97072 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97073 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_12 := arithEq_of_rows h (row := 1584) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_12
  simp only [← c97068, ← c97069, ← c97070] at e_1584_12
  have e_1584_13 := arithEq_of_rows h (row := 1584) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_13
  simp only [← c97071, ← c97072, k2, ← c97073] at e_1584_13
  have hr := e_1584_13
  simp only [e_1584_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15492)) := by
  have c97074 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97075 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97076 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97077 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97078 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97079 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_14 := arithEq_of_rows h (row := 1584) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_14
  simp only [← c97074, ← c97075, ← c97076] at e_1584_14
  have e_1584_15 := arithEq_of_rows h (row := 1584) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_15
  simp only [← c97077, ← c97078, k2, ← c97079] at e_1584_15
  have hr := e_1584_15
  simp only [e_1584_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15493)) := by
  have c97080 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97081 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97082 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97083 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97084 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97085 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_16 := arithEq_of_rows h (row := 1584) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_16
  simp only [← c97080, ← c97081, ← c97082] at e_1584_16
  have e_1584_17 := arithEq_of_rows h (row := 1584) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_17
  simp only [← c97083, ← c97084, k2, ← c97085] at e_1584_17
  have hr := e_1584_17
  simp only [e_1584_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15494)) := by
  have c97086 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97087 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97088 := (publicBatchWrapper89_copies3034 a h).1
  have c97089 := (publicBatchWrapper89_copies3034 a h).2.1
  have c97090 := (publicBatchWrapper89_copies3034 a h).2.2.1
  have c97091 := (publicBatchWrapper89_copies3034 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_18 := arithEq_of_rows h (row := 1584) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_18
  simp only [← c97086, ← c97087, ← c97088] at e_1584_18
  have e_1584_19 := arithEq_of_rows h (row := 1584) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_19
  simp only [← c97089, ← c97090, k2, ← c97091] at e_1584_19
  have hr := e_1584_19
  simp only [e_1584_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15495)) := by
  have c97092 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.1
  have c97093 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.1
  have c97094 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.1
  have c97095 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.1
  have c97096 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.1
  have c97097 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_0 := arithEq_of_rows h (row := 1585) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_0
  simp only [← c97092, ← c97093, ← c97094] at e_1585_0
  have e_1585_1 := arithEq_of_rows h (row := 1585) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_1
  simp only [← c97095, ← c97096, k2, ← c97097] at e_1585_1
  have hr := e_1585_1
  simp only [e_1585_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15496)) := by
  have c97098 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97099 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97100 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97101 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97102 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97103 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_2 := arithEq_of_rows h (row := 1585) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_2
  simp only [← c97098, ← c97099, ← c97100] at e_1585_2
  have e_1585_3 := arithEq_of_rows h (row := 1585) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_3
  simp only [← c97101, ← c97102, k2, ← c97103] at e_1585_3
  have hr := e_1585_3
  simp only [e_1585_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15497)) := by
  have c97104 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97105 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97106 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97107 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97108 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97109 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_4 := arithEq_of_rows h (row := 1585) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_4
  simp only [← c97104, ← c97105, ← c97106] at e_1585_4
  have e_1585_5 := arithEq_of_rows h (row := 1585) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_5
  simp only [← c97107, ← c97108, k2, ← c97109] at e_1585_5
  have hr := e_1585_5
  simp only [e_1585_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15498)) := by
  have c97110 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97111 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97112 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97113 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97114 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97115 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_6 := arithEq_of_rows h (row := 1585) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_6
  simp only [← c97110, ← c97111, ← c97112] at e_1585_6
  have e_1585_7 := arithEq_of_rows h (row := 1585) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_7
  simp only [← c97113, ← c97114, k2, ← c97115] at e_1585_7
  have hr := e_1585_7
  simp only [e_1585_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15499)) := by
  have c97116 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97117 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97118 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97119 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97120 := (publicBatchWrapper89_copies3035 a h).1
  have c97121 := (publicBatchWrapper89_copies3035 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_8 := arithEq_of_rows h (row := 1585) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_8
  simp only [← c97116, ← c97117, ← c97118] at e_1585_8
  have e_1585_9 := arithEq_of_rows h (row := 1585) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_9
  simp only [← c97119, ← c97120, k2, ← c97121] at e_1585_9
  have hr := e_1585_9
  simp only [e_1585_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
