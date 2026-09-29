/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from the gadget-zoo circuit built through the recording builder: the
  pre-`build` constraint system and the gadget calls recorded while building it.
  Each theorem's proof is generated too, one block per recorded gadget call, from
  the ops and copy constraints the builder emitted for it.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

/-- `gadgetZoo.copies`, items `0..32`. -/
def gadgetZoo.copies0 : List (Target × Target) := [
    (.virt 3, .wire 0 0),
    (.virt 3, .wire 0 1),
    (.virt 3, .wire 0 2),
    (.wire 0 3, .virt 4),
    (.virt 6, .wire 0 4),
    (.virt 6, .wire 0 5),
    (.virt 5, .wire 0 6),
    (.virt 0, .wire 0 8),
    (.virt 6, .wire 0 9),
    (.virt 1, .wire 0 10),
    (.virt 5, .wire 1 0),
    (.wire 0 11, .wire 1 1),
    (.virt 5, .wire 1 2),
    (.wire 0 11, .wire 1 4),
    (.virt 7, .wire 1 5),
    (.wire 0 11, .wire 1 6),
    (.wire 1 7, .wire 0 12),
    (.virt 6, .wire 0 13),
    (.wire 0 7, .wire 0 14),
    (.wire 1 3, .virt 4),
    (.wire 0 15, .virt 4),
    (.virt 3, .wire 0 16),
    (.virt 1, .wire 0 17),
    (.virt 1, .wire 0 18),
    (.virt 3, .wire 0 20),
    (.virt 0, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.virt 5, .wire 2 0),
    (.virt 3, .wire 2 1),
    (.virt 5, .wire 2 2),
    (.wire 2 3, .wire 3 0),
    (.virt 6, .wire 3 1)
  ]

/-- `gadgetZoo.copies`, items `32..64`. -/
def gadgetZoo.copies1 : List (Target × Target) := [
    (.virt 3, .wire 3 2),
    (.virt 6, .wire 0 24),
    (.virt 6, .wire 0 25),
    (.virt 3, .wire 0 26),
    (.virt 5, .wire 1 8),
    (.wire 0 27, .wire 1 9),
    (.virt 5, .wire 1 10),
    (.virt 8, .wire 0 28),
    (.virt 6, .wire 0 29),
    (.virt 2, .wire 0 30),
    (.wire 4 15, .virt 4),
    (.wire 4 16, .virt 4),
    (.wire 4 17, .virt 4),
    (.wire 4 18, .virt 4),
    (.wire 4 19, .virt 4),
    (.wire 4 20, .virt 4),
    (.wire 4 21, .virt 4),
    (.wire 4 22, .virt 4),
    (.wire 4 23, .virt 4),
    (.wire 4 24, .virt 4),
    (.wire 4 25, .virt 4),
    (.wire 4 26, .virt 4),
    (.wire 4 27, .virt 4),
    (.wire 4 28, .virt 4),
    (.wire 4 29, .virt 4),
    (.wire 4 30, .virt 4),
    (.wire 4 31, .virt 4),
    (.wire 4 32, .virt 4),
    (.wire 4 33, .virt 4),
    (.wire 4 34, .virt 4),
    (.wire 4 35, .virt 4),
    (.wire 4 36, .virt 4)
  ]

/-- `gadgetZoo.copies`, items `64..93`. -/
def gadgetZoo.copies2 : List (Target × Target) := [
    (.wire 4 37, .virt 4),
    (.wire 4 38, .virt 4),
    (.wire 4 39, .virt 4),
    (.wire 4 40, .virt 4),
    (.wire 4 41, .virt 4),
    (.wire 4 42, .virt 4),
    (.wire 4 43, .virt 4),
    (.wire 4 44, .virt 4),
    (.wire 4 45, .virt 4),
    (.wire 4 46, .virt 4),
    (.wire 4 47, .virt 4),
    (.wire 4 48, .virt 4),
    (.wire 4 49, .virt 4),
    (.wire 4 50, .virt 4),
    (.wire 4 51, .virt 4),
    (.wire 4 52, .virt 4),
    (.wire 4 53, .virt 4),
    (.wire 4 54, .virt 4),
    (.wire 4 55, .virt 4),
    (.wire 4 56, .virt 4),
    (.wire 4 57, .virt 4),
    (.wire 4 58, .virt 4),
    (.wire 4 59, .virt 4),
    (.wire 4 60, .virt 4),
    (.wire 4 61, .virt 4),
    (.wire 4 62, .virt 4),
    (.wire 4 63, .virt 4),
    (.wire 4 0, .wire 0 31),
    (.wire 0 23, .wire 3 3)
  ]

/-- The gadget zoo: `assert_bool flag`, `eq = is_equal x y`, `sel = select flag x y`, `either = or eq flag`, `nflag = not flag`, `both = and eq nflag`, `head = sub 10000 fee`, `range_check head 14`, `connect sel either`; public inputs `x`, `sel`, `both`. -/
def gadgetZoo (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 20, [(-1), 1]⟩,  -- row 2
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 3
    ⟨.baseSum2 63, []⟩  -- row 4
  ]
  copies := (gadgetZoo.copies0 ++ (gadgetZoo.copies1 ++ gadgetZoo.copies2))
  constants := [
    (.virt 4, 0),
    (.virt 6, 1),
    (.virt 8, 10000),
    (.virt 9, 9223372036854775808 /- canonical u64; faithful only at p = goldilocks -/)
  ]
  publicInputs := [
    .virt 0,
    .wire 0 23,
    .wire 1 11
  ]

/-- Named target `x`. -/
def gadgetZoo.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetZoo.y : Target := .virt 1

/-- Named target `fee`. -/
def gadgetZoo.fee : Target := .virt 2

/-- Named target `flag`. -/
def gadgetZoo.flag : Target := .virt 3

/-- Named target `eq`. -/
def gadgetZoo.eq : Target := .virt 5

/-- Named target `sel`. -/
def gadgetZoo.sel : Target := .wire 0 23

/-- Named target `head`. -/
def gadgetZoo.head : Target := .wire 0 31

theorem gadgetZoo_copies0 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.virt 3) = a (.wire 0 0) ∧ a (.virt 3) = a (.wire 0 1) ∧ a (.virt 3) = a (.wire 0 2) ∧ a (.wire 0 3) = a (.virt 4) ∧ a (.virt 6) = a (.wire 0 4) ∧ a (.virt 6) = a (.wire 0 5) ∧ a (.virt 5) = a (.wire 0 6) ∧ a (.virt 0) = a (.wire 0 8) ∧ a (.virt 6) = a (.wire 0 9) ∧ a (.virt 1) = a (.wire 0 10) ∧ a (.virt 5) = a (.wire 1 0) ∧ a (.wire 0 11) = a (.wire 1 1) ∧ a (.virt 5) = a (.wire 1 2) ∧ a (.wire 0 11) = a (.wire 1 4) ∧ a (.virt 7) = a (.wire 1 5) ∧ a (.wire 0 11) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 12) ∧ a (.virt 6) = a (.wire 0 13) ∧ a (.wire 0 7) = a (.wire 0 14) ∧ a (.wire 1 3) = a (.virt 4) ∧ a (.wire 0 15) = a (.virt 4) ∧ a (.virt 3) = a (.wire 0 16) ∧ a (.virt 1) = a (.wire 0 17) ∧ a (.virt 1) = a (.wire 0 18) ∧ a (.virt 3) = a (.wire 0 20) ∧ a (.virt 0) = a (.wire 0 21) ∧ a (.wire 0 19) = a (.wire 0 22) ∧ a (.virt 5) = a (.wire 2 0) ∧ a (.virt 3) = a (.wire 2 1) ∧ a (.virt 5) = a (.wire 2 2) ∧ a (.wire 2 3) = a (.wire 3 0) ∧ a (.virt 6) = a (.wire 3 1) := by
  have hc : ∀ q ∈ gadgetZoo.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ hq)
  simp only [gadgetZoo.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetZoo_copies1 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.virt 3) = a (.wire 3 2) ∧ a (.virt 6) = a (.wire 0 24) ∧ a (.virt 6) = a (.wire 0 25) ∧ a (.virt 3) = a (.wire 0 26) ∧ a (.virt 5) = a (.wire 1 8) ∧ a (.wire 0 27) = a (.wire 1 9) ∧ a (.virt 5) = a (.wire 1 10) ∧ a (.virt 8) = a (.wire 0 28) ∧ a (.virt 6) = a (.wire 0 29) ∧ a (.virt 2) = a (.wire 0 30) ∧ a (.wire 4 15) = a (.virt 4) ∧ a (.wire 4 16) = a (.virt 4) ∧ a (.wire 4 17) = a (.virt 4) ∧ a (.wire 4 18) = a (.virt 4) ∧ a (.wire 4 19) = a (.virt 4) ∧ a (.wire 4 20) = a (.virt 4) ∧ a (.wire 4 21) = a (.virt 4) ∧ a (.wire 4 22) = a (.virt 4) ∧ a (.wire 4 23) = a (.virt 4) ∧ a (.wire 4 24) = a (.virt 4) ∧ a (.wire 4 25) = a (.virt 4) ∧ a (.wire 4 26) = a (.virt 4) ∧ a (.wire 4 27) = a (.virt 4) ∧ a (.wire 4 28) = a (.virt 4) ∧ a (.wire 4 29) = a (.virt 4) ∧ a (.wire 4 30) = a (.virt 4) ∧ a (.wire 4 31) = a (.virt 4) ∧ a (.wire 4 32) = a (.virt 4) ∧ a (.wire 4 33) = a (.virt 4) ∧ a (.wire 4 34) = a (.virt 4) ∧ a (.wire 4 35) = a (.virt 4) ∧ a (.wire 4 36) = a (.virt 4) := by
  have hc : ∀ q ∈ gadgetZoo.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ hq))
  simp only [gadgetZoo.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetZoo_copies2 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 4 37) = a (.virt 4) ∧ a (.wire 4 38) = a (.virt 4) ∧ a (.wire 4 39) = a (.virt 4) ∧ a (.wire 4 40) = a (.virt 4) ∧ a (.wire 4 41) = a (.virt 4) ∧ a (.wire 4 42) = a (.virt 4) ∧ a (.wire 4 43) = a (.virt 4) ∧ a (.wire 4 44) = a (.virt 4) ∧ a (.wire 4 45) = a (.virt 4) ∧ a (.wire 4 46) = a (.virt 4) ∧ a (.wire 4 47) = a (.virt 4) ∧ a (.wire 4 48) = a (.virt 4) ∧ a (.wire 4 49) = a (.virt 4) ∧ a (.wire 4 50) = a (.virt 4) ∧ a (.wire 4 51) = a (.virt 4) ∧ a (.wire 4 52) = a (.virt 4) ∧ a (.wire 4 53) = a (.virt 4) ∧ a (.wire 4 54) = a (.virt 4) ∧ a (.wire 4 55) = a (.virt 4) ∧ a (.wire 4 56) = a (.virt 4) ∧ a (.wire 4 57) = a (.virt 4) ∧ a (.wire 4 58) = a (.virt 4) ∧ a (.wire 4 59) = a (.virt 4) ∧ a (.wire 4 60) = a (.virt 4) ∧ a (.wire 4 61) = a (.virt 4) ∧ a (.wire 4 62) = a (.virt 4) ∧ a (.wire 4 63) = a (.virt 4) ∧ a (.wire 4 0) = a (.wire 0 31) ∧ a (.wire 0 23) = a (.wire 3 3) := by
  have hc : ∀ q ∈ gadgetZoo.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ hq))
  simp only [gadgetZoo.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetZoo_consts (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.virt 4) = 0 ∧ a (.virt 6) = 1 ∧ a (.virt 8) = 10000 ∧ a (.virt 9) = 9223372036854775808 := by
  have hconst := h.2.2
  simp only [gadgetZoo, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3⟩ := hconst
  exact ⟨k0, k1, k2, k3⟩

theorem gadgetZoo_f0 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    IsBool (a (.virt 3)) := by
  have c0 := (gadgetZoo_copies0 a h).1
  have c1 := (gadgetZoo_copies0 a h).2.1
  have c2 := (gadgetZoo_copies0 a h).2.2.1
  have c3 := (gadgetZoo_copies0 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := gadgetZoo_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, ← c2] at e_0_0
  refine isBool_iff_assertBool.mpr ?_
  have hc := e_0_0
  linear_combination c3.trans k0 - hc

theorem gadgetZoo_f1 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 5)) (a (.virt 7)) := by
  have c4 := (gadgetZoo_copies0 a h).2.2.2.2.1
  have c5 := (gadgetZoo_copies0 a h).2.2.2.2.2.1
  have c6 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.1
  have c7 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := gadgetZoo_consts a h
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_1
  simp only [← c4, k1, ← c5, k1, ← c6] at e_0_1
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_2
  simp only [← c7, ← c8, k1, ← c9] at e_0_2
  have e_0_3 := arithEq_of_rows h (row := 0) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_3
  simp only [← c16, ← c17, k1, ← c18] at e_0_3
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_0
  simp only [← c10, ← c11, ← c12] at e_1_0
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_1
  simp only [← c13, ← c14, ← c15] at e_1_1
  refine ⟨?_, ?_⟩
  · have hc := e_1_0
    simp only [e_0_2] at hc
    linear_combination c19.trans k0 - hc
  · have hc := e_0_3
    simp only [e_0_1, e_1_1, e_0_2] at hc
    linear_combination c20.trans k0 - hc

theorem gadgetZoo_f2 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 0 23) = bselect (a (.virt 3)) (a (.virt 0)) (a (.virt 1)) := by
  have c21 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_0_4 := arithEq_of_rows h (row := 0) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_4
  simp only [← c21, ← c22, ← c23] at e_0_4
  have e_0_5 := arithEq_of_rows h (row := 0) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_5
  simp only [← c24, ← c25, ← c26] at e_0_5
  have hr := e_0_5
  simp only [e_0_4] at hr
  simp only [bselect]
  linear_combination hr

theorem gadgetZoo_f3 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 3 3) = bor (a (.virt 5)) (a (.virt 3)) := by
  have c27 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (gadgetZoo_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (gadgetZoo_copies1 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := gadgetZoo_consts a h
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c27, ← c28, ← c29] at e_2_0
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c30, ← c31, k1, ← c32] at e_3_0
  have hr := e_3_0
  simp only [e_2_0] at hr
  simp only [bor]
  linear_combination hr

theorem gadgetZoo_f4 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 0 27) = bnot (a (.virt 3)) := by
  have c33 := (gadgetZoo_copies1 a h).2.1
  have c34 := (gadgetZoo_copies1 a h).2.2.1
  have c35 := (gadgetZoo_copies1 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := gadgetZoo_consts a h
  have e_0_6 := arithEq_of_rows h (row := 0) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_6
  simp only [← c33, k1, ← c34, k1, ← c35] at e_0_6
  have hr := e_0_6
  simp only [bnot]
  linear_combination hr

theorem gadgetZoo_f5 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 1 11) = band (a (.virt 5)) (a (.wire 0 27)) := by
  have c36 := (gadgetZoo_copies1 a h).2.2.2.2.1
  have c37 := (gadgetZoo_copies1 a h).2.2.2.2.2.1
  have c38 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.1
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_2
  simp only [← c36, ← c37, ← c38] at e_1_2
  have hr := e_1_2
  simp only [band]
  linear_combination hr

theorem gadgetZoo_f6 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 0 31) = a (.virt 8) - a (.virt 2) := by
  have c39 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := gadgetZoo_consts a h
  have e_0_7 := arithEq_of_rows h (row := 0) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_7
  simp only [← c39, k2, ← c40, k1, ← c41] at e_0_7
  have hr := e_0_7
  simp only [k2]
  linear_combination hr

theorem gadgetZoo_f7 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    rangeCheck (a (.wire 0 31)) 14 := by
  have c42 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (gadgetZoo_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (gadgetZoo_copies2 a h).1
  have c65 := (gadgetZoo_copies2 a h).2.1
  have c66 := (gadgetZoo_copies2 a h).2.2.1
  have c67 := (gadgetZoo_copies2 a h).2.2.2.1
  have c68 := (gadgetZoo_copies2 a h).2.2.2.2.1
  have c69 := (gadgetZoo_copies2 a h).2.2.2.2.2.1
  have c70 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.1
  have c71 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := gadgetZoo_consts a h
  have hr := rangeCheck_of_row h (row := 4) (N := 63) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c42.trans k0
    · exact c43.trans k0
    · exact c44.trans k0
    · exact c45.trans k0
    · exact c46.trans k0
    · exact c47.trans k0
    · exact c48.trans k0
    · exact c49.trans k0
    · exact c50.trans k0
    · exact c51.trans k0
    · exact c52.trans k0
    · exact c53.trans k0
    · exact c54.trans k0
    · exact c55.trans k0
    · exact c56.trans k0
    · exact c57.trans k0
    · exact c58.trans k0
    · exact c59.trans k0
    · exact c60.trans k0
    · exact c61.trans k0
    · exact c62.trans k0
    · exact c63.trans k0
    · exact c64.trans k0
    · exact c65.trans k0
    · exact c66.trans k0
    · exact c67.trans k0
    · exact c68.trans k0
    · exact c69.trans k0
    · exact c70.trans k0
    · exact c71.trans k0
    · exact c72.trans k0
    · exact c73.trans k0
    · exact c74.trans k0
    · exact c75.trans k0
    · exact c76.trans k0
    · exact c77.trans k0
    · exact c78.trans k0
    · exact c79.trans k0
    · exact c80.trans k0
    · exact c81.trans k0
    · exact c82.trans k0
    · exact c83.trans k0
    · exact c84.trans k0
    · exact c85.trans k0
    · exact c86.trans k0
    · exact c87.trans k0
    · exact c88.trans k0
    · exact c89.trans k0
    · exact c90.trans k0
    )
  rwa [c91] at hr

theorem gadgetZoo_f8 (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    a (.wire 0 23) = a (.wire 3 3) := by
  have c92 := (gadgetZoo_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c92

/-- Every satisfying assignment of `gadgetZoo` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetZoo_decode (a : Assignment p) (h : Satisfies (gadgetZoo p) a) :
    IsBool (a (.virt 3)) ∧
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 5)) (a (.virt 7)) ∧
    a (.wire 0 23) = bselect (a (.virt 3)) (a (.virt 0)) (a (.virt 1)) ∧
    a (.wire 3 3) = bor (a (.virt 5)) (a (.virt 3)) ∧
    a (.wire 0 27) = bnot (a (.virt 3)) ∧
    a (.wire 1 11) = band (a (.virt 5)) (a (.wire 0 27)) ∧
    a (.wire 0 31) = a (.virt 8) - a (.virt 2) ∧
    rangeCheck (a (.wire 0 31)) 14 ∧
    a (.wire 0 23) = a (.wire 3 3) :=
  ⟨gadgetZoo_f0 a h, gadgetZoo_f1 a h, gadgetZoo_f2 a h, gadgetZoo_f3 a h, gadgetZoo_f4 a h, gadgetZoo_f5 a h, gadgetZoo_f6 a h, gadgetZoo_f7 a h, gadgetZoo_f8 a h⟩

end Plonky2Spec.Generated
