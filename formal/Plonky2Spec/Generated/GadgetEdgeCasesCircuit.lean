/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from the gadget edge-case circuits built through the recording builder: the
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

/-- Builder folding and re-pinning edge cases: `sum = add x y; connect sum zero`, `is_equal zero zero`, `is_equal one zero`, `is_equal x x`, `eq_xy = is_equal x y; diff = sub x y; check = mul eq_xy diff; connect check zero`; public inputs the four `equal` targets. -/
def gadgetEdgeCases (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 0
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 1
    ⟨.arithmetic 20, [1, 0]⟩  -- row 2
  ]
  copies := [
    (.virt 0, .wire 0 0),
    (.virt 3, .wire 0 1),
    (.virt 1, .wire 0 2),
    (.wire 0 3, .virt 2),
    (.virt 3, .wire 1 0),
    (.virt 3, .wire 1 1),
    (.virt 4, .wire 1 2),
    (.virt 2, .wire 1 4),
    (.virt 3, .wire 1 5),
    (.wire 1 3, .wire 1 6),
    (.virt 2, .virt 2),
    (.wire 1 7, .virt 2),
    (.virt 3, .wire 1 8),
    (.virt 3, .wire 1 9),
    (.virt 6, .wire 1 10),
    (.virt 7, .wire 1 12),
    (.virt 3, .wire 1 13),
    (.wire 1 11, .wire 1 14),
    (.virt 6, .virt 2),
    (.wire 1 15, .virt 2),
    (.virt 3, .wire 1 16),
    (.virt 3, .wire 1 17),
    (.virt 8, .wire 1 18),
    (.virt 0, .wire 1 20),
    (.virt 3, .wire 1 21),
    (.virt 0, .wire 1 22),
    (.virt 8, .wire 2 0),
    (.wire 1 23, .wire 2 1),
    (.virt 8, .wire 2 2),
    (.wire 1 23, .wire 2 4),
    (.virt 9, .wire 2 5),
    (.wire 1 23, .wire 2 6),
    (.wire 2 7, .wire 1 24),
    (.virt 3, .wire 1 25),
    (.wire 1 19, .wire 1 26),
    (.wire 2 3, .virt 2),
    (.wire 1 27, .virt 2),
    (.virt 3, .wire 1 28),
    (.virt 3, .wire 1 29),
    (.virt 10, .wire 1 30),
    (.virt 0, .wire 1 32),
    (.virt 3, .wire 1 33),
    (.virt 1, .wire 1 34),
    (.virt 10, .wire 2 8),
    (.wire 1 35, .wire 2 9),
    (.virt 10, .wire 2 10),
    (.wire 1 35, .wire 2 12),
    (.virt 11, .wire 2 13),
    (.wire 1 35, .wire 2 14),
    (.wire 2 15, .wire 1 36),
    (.virt 3, .wire 1 37),
    (.wire 1 31, .wire 1 38),
    (.wire 2 11, .virt 2),
    (.wire 1 39, .virt 2),
    (.wire 2 11, .virt 2)
  ]
  constants := [
    (.virt 2, 0),
    (.virt 3, 1)
  ]
  publicInputs := [
    .virt 4,
    .virt 6,
    .virt 8,
    .virt 10
  ]

/-- Named target `x`. -/
def gadgetEdgeCases.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetEdgeCases.y : Target := .virt 1

/-- Named target `zero`. -/
def gadgetEdgeCases.zero : Target := .virt 2

/-- Named target `one`. -/
def gadgetEdgeCases.one : Target := .virt 3

/-- Named target `sum`. -/
def gadgetEdgeCases.sum : Target := .wire 0 3

/-- Named target `eq_zz`. -/
def gadgetEdgeCases.eq_zz : Target := .virt 4

/-- Named target `eq_oz`. -/
def gadgetEdgeCases.eq_oz : Target := .virt 6

/-- Named target `eq_xx`. -/
def gadgetEdgeCases.eq_xx : Target := .virt 8

/-- Named target `eq_xy`. -/
def gadgetEdgeCases.eq_xy : Target := .virt 10

/-- Named target `diff`. -/
def gadgetEdgeCases.diff : Target := .wire 1 35

/-- Named target `check`. -/
def gadgetEdgeCases.check : Target := .wire 2 11

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetEdgeCases.verifiers : List (String × List Target) :=
  []

theorem gadgetEdgeCases_copies (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.virt 0) = a (.wire 0 0) ∧ a (.virt 3) = a (.wire 0 1) ∧ a (.virt 1) = a (.wire 0 2) ∧ a (.wire 0 3) = a (.virt 2) ∧ a (.virt 3) = a (.wire 1 0) ∧ a (.virt 3) = a (.wire 1 1) ∧ a (.virt 4) = a (.wire 1 2) ∧ a (.virt 2) = a (.wire 1 4) ∧ a (.virt 3) = a (.wire 1 5) ∧ a (.wire 1 3) = a (.wire 1 6) ∧ True ∧ a (.wire 1 7) = a (.virt 2) ∧ a (.virt 3) = a (.wire 1 8) ∧ a (.virt 3) = a (.wire 1 9) ∧ a (.virt 6) = a (.wire 1 10) ∧ a (.virt 7) = a (.wire 1 12) ∧ a (.virt 3) = a (.wire 1 13) ∧ a (.wire 1 11) = a (.wire 1 14) ∧ a (.virt 6) = a (.virt 2) ∧ a (.wire 1 15) = a (.virt 2) ∧ a (.virt 3) = a (.wire 1 16) ∧ a (.virt 3) = a (.wire 1 17) ∧ a (.virt 8) = a (.wire 1 18) ∧ a (.virt 0) = a (.wire 1 20) ∧ a (.virt 3) = a (.wire 1 21) ∧ a (.virt 0) = a (.wire 1 22) ∧ a (.virt 8) = a (.wire 2 0) ∧ a (.wire 1 23) = a (.wire 2 1) ∧ a (.virt 8) = a (.wire 2 2) ∧ a (.wire 1 23) = a (.wire 2 4) ∧ a (.virt 9) = a (.wire 2 5) ∧ a (.wire 1 23) = a (.wire 2 6) ∧ a (.wire 2 7) = a (.wire 1 24) ∧ a (.virt 3) = a (.wire 1 25) ∧ a (.wire 1 19) = a (.wire 1 26) ∧ a (.wire 2 3) = a (.virt 2) ∧ a (.wire 1 27) = a (.virt 2) ∧ a (.virt 3) = a (.wire 1 28) ∧ a (.virt 3) = a (.wire 1 29) ∧ a (.virt 10) = a (.wire 1 30) ∧ a (.virt 0) = a (.wire 1 32) ∧ a (.virt 3) = a (.wire 1 33) ∧ a (.virt 1) = a (.wire 1 34) ∧ a (.virt 10) = a (.wire 2 8) ∧ a (.wire 1 35) = a (.wire 2 9) ∧ a (.virt 10) = a (.wire 2 10) ∧ a (.wire 1 35) = a (.wire 2 12) ∧ a (.virt 11) = a (.wire 2 13) ∧ a (.wire 1 35) = a (.wire 2 14) ∧ a (.wire 2 15) = a (.wire 1 36) ∧ a (.virt 3) = a (.wire 1 37) ∧ a (.wire 1 31) = a (.wire 1 38) ∧ a (.wire 2 11) = a (.virt 2) ∧ a (.wire 1 39) = a (.virt 2) ∧ a (.wire 2 11) = a (.virt 2) := by
  have hc := h.2.1
  simp only [gadgetEdgeCases, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetEdgeCases_consts (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.virt 2) = 0 ∧ a (.virt 3) = 1 := by
  have hconst := h.2.2
  simp only [gadgetEdgeCases, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1⟩ := hconst
  exact ⟨k0, k1⟩

theorem gadgetEdgeCases_f0 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
  have c0 := (gadgetEdgeCases_copies a h).1
  have c1 := (gadgetEdgeCases_copies a h).2.1
  have c2 := (gadgetEdgeCases_copies a h).2.2.1
  obtain ⟨k0, k1⟩ := gadgetEdgeCases_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, k1, ← c2] at e_0_0
  have hr := e_0_0
  linear_combination hr

theorem gadgetEdgeCases_f1 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.wire 0 3) = a (.virt 2) := by
  have c3 := (gadgetEdgeCases_copies a h).2.2.2.1
  exact c3

theorem gadgetEdgeCases_f2 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    IsEqual (a (.virt 2)) (a (.virt 2)) (a (.virt 4)) (a (.virt 5)) := by
  have c4 := (gadgetEdgeCases_copies a h).2.2.2.2.1
  have c5 := (gadgetEdgeCases_copies a h).2.2.2.2.2.1
  have c6 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.1
  have c7 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.1
  have c8 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.1
  have c9 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.1
  have c11 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1⟩ := gadgetEdgeCases_consts a h
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_0
  simp only [← c4, k1, ← c5, k1, ← c6] at e_1_0
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_1
  simp only [← c7, k0, ← c8, k1, ← c9] at e_1_1
  simp only [k0]
  refine ⟨?_, ?_⟩
  · ring
  · have hc := e_1_1
    simp only [e_1_0] at hc
    linear_combination c11.trans k0 - hc

theorem gadgetEdgeCases_f3 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    IsEqual (a (.virt 3)) (a (.virt 2)) (a (.virt 6)) (a (.virt 7)) := by
  have c12 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1⟩ := gadgetEdgeCases_consts a h
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_2
  simp only [← c12, k1, ← c13, k1, ← c14] at e_1_2
  have e_1_3 := arithEq_of_rows h (row := 1) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_3
  simp only [← c15, ← c16, k1, ← c17] at e_1_3
  simp only [k0, k1]
  refine ⟨?_, ?_⟩
  · rw [c18.trans k0]
    ring
  · have hc := e_1_3
    simp only [e_1_2] at hc
    linear_combination c19.trans k0 - hc

theorem gadgetEdgeCases_f4 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    IsEqual (a (.virt 0)) (a (.virt 0)) (a (.virt 8)) (a (.virt 9)) := by
  have c20 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1⟩ := gadgetEdgeCases_consts a h
  have e_1_4 := arithEq_of_rows h (row := 1) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_4
  simp only [← c20, k1, ← c21, k1, ← c22] at e_1_4
  have e_1_5 := arithEq_of_rows h (row := 1) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_5
  simp only [← c23, ← c24, k1, ← c25] at e_1_5
  have e_1_6 := arithEq_of_rows h (row := 1) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_6
  simp only [← c32, ← c33, k1, ← c34] at e_1_6
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c26, ← c27, ← c28] at e_2_0
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c29, ← c30, ← c31] at e_2_1
  refine ⟨?_, ?_⟩
  · have hc := e_2_0
    simp only [e_1_5] at hc
    linear_combination c35.trans k0 - hc
  · have hc := e_1_6
    simp only [e_1_4, e_2_1, e_1_5] at hc
    linear_combination c36.trans k0 - hc

theorem gadgetEdgeCases_f5 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 10)) (a (.virt 11)) := by
  have c37 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1⟩ := gadgetEdgeCases_consts a h
  have e_1_7 := arithEq_of_rows h (row := 1) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_7
  simp only [← c37, k1, ← c38, k1, ← c39] at e_1_7
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c40, ← c41, k1, ← c42] at e_1_8
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c49, ← c50, k1, ← c51] at e_1_9
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c43, ← c44, ← c45] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c46, ← c47, ← c48] at e_2_3
  refine ⟨?_, ?_⟩
  · have hc := e_2_2
    simp only [e_1_8] at hc
    linear_combination c52.trans k0 - hc
  · have hc := e_1_9
    simp only [e_1_7, e_2_3, e_1_8] at hc
    linear_combination c53.trans k0 - hc

theorem gadgetEdgeCases_f6 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.wire 1 35) = a (.virt 0) - a (.virt 1) := by
  have c40 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1⟩ := gadgetEdgeCases_consts a h
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c40, ← c41, k1, ← c42] at e_1_8
  have hr := e_1_8
  linear_combination hr

theorem gadgetEdgeCases_f7 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.wire 2 11) = a (.virt 10) * a (.wire 1 35) := by
  have c43 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c43, ← c44, ← c45] at e_2_2
  have hr := e_2_2
  linear_combination hr

theorem gadgetEdgeCases_f8 (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.wire 2 11) = a (.virt 2) := by
  have c54 := (gadgetEdgeCases_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c54

/-- Every satisfying assignment of `gadgetEdgeCases` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetEdgeCases_decode (a : Assignment p) (h : Satisfies (gadgetEdgeCases p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) ∧
    a (.wire 0 3) = a (.virt 2) ∧
    IsEqual (a (.virt 2)) (a (.virt 2)) (a (.virt 4)) (a (.virt 5)) ∧
    IsEqual (a (.virt 3)) (a (.virt 2)) (a (.virt 6)) (a (.virt 7)) ∧
    IsEqual (a (.virt 0)) (a (.virt 0)) (a (.virt 8)) (a (.virt 9)) ∧
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 10)) (a (.virt 11)) ∧
    a (.wire 1 35) = a (.virt 0) - a (.virt 1) ∧
    a (.wire 2 11) = a (.virt 10) * a (.wire 1 35) ∧
    a (.wire 2 11) = a (.virt 2) :=
  ⟨gadgetEdgeCases_f0 a h, gadgetEdgeCases_f1 a h, gadgetEdgeCases_f2 a h, gadgetEdgeCases_f3 a h, gadgetEdgeCases_f4 a h, gadgetEdgeCases_f5 a h, gadgetEdgeCases_f6 a h, gadgetEdgeCases_f7 a h, gadgetEdgeCases_f8 a h⟩

/-- `sum = add x y; prod = mul sum one`: the product folds onto `sum`. -/
def gadgetIdentityFold (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, 1]⟩  -- row 0
  ]
  copies := [
    (.virt 0, .wire 0 0),
    (.virt 2, .wire 0 1),
    (.virt 1, .wire 0 2)
  ]
  constants := [
    (.virt 3, 0),
    (.virt 2, 1)
  ]
  publicInputs := [
  ]

/-- Named target `x`. -/
def gadgetIdentityFold.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetIdentityFold.y : Target := .virt 1

/-- Named target `one`. -/
def gadgetIdentityFold.one : Target := .virt 2

/-- Named target `sum`. -/
def gadgetIdentityFold.sum : Target := .wire 0 3

/-- Named target `prod`. -/
def gadgetIdentityFold.prod : Target := .wire 0 3

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetIdentityFold.verifiers : List (String × List Target) :=
  []

theorem gadgetIdentityFold_copies (a : Assignment p) (h : Satisfies (gadgetIdentityFold p) a) :
    a (.virt 0) = a (.wire 0 0) ∧ a (.virt 2) = a (.wire 0 1) ∧ a (.virt 1) = a (.wire 0 2) := by
  have hc := h.2.1
  simp only [gadgetIdentityFold, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetIdentityFold_consts (a : Assignment p) (h : Satisfies (gadgetIdentityFold p) a) :
    a (.virt 3) = 0 ∧ a (.virt 2) = 1 := by
  have hconst := h.2.2
  simp only [gadgetIdentityFold, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1⟩ := hconst
  exact ⟨k0, k1⟩

theorem gadgetIdentityFold_f0 (a : Assignment p) (h : Satisfies (gadgetIdentityFold p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
  have c0 := (gadgetIdentityFold_copies a h).1
  have c1 := (gadgetIdentityFold_copies a h).2.1
  have c2 := (gadgetIdentityFold_copies a h).2.2
  obtain ⟨k0, k1⟩ := gadgetIdentityFold_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, k1, ← c2] at e_0_0
  have hr := e_0_0
  linear_combination hr

theorem gadgetIdentityFold_f1 (a : Assignment p) (h : Satisfies (gadgetIdentityFold p) a) :
    a (.wire 0 3) = a (.wire 0 3) * a (.virt 2) := by
  obtain ⟨k0, k1⟩ := gadgetIdentityFold_consts a h
  simp only [k1]
  ring

/-- Every satisfying assignment of `gadgetIdentityFold` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetIdentityFold_decode (a : Assignment p) (h : Satisfies (gadgetIdentityFold p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) ∧
    a (.wire 0 3) = a (.wire 0 3) * a (.virt 2) :=
  ⟨gadgetIdentityFold_f0 a h, gadgetIdentityFold_f1 a h⟩

/-- `diff = sub x y; eq = is_equal x y; connect diff zero`: `is_equal` reuses `diff`, which is then pinned to zero. -/
def gadgetPinnedIntermediate (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 20, [1, 0]⟩  -- row 1
  ]
  copies := [
    (.virt 0, .wire 0 0),
    (.virt 3, .wire 0 1),
    (.virt 1, .wire 0 2),
    (.virt 3, .wire 0 4),
    (.virt 3, .wire 0 5),
    (.virt 4, .wire 0 6),
    (.virt 4, .wire 1 0),
    (.wire 0 3, .wire 1 1),
    (.virt 4, .wire 1 2),
    (.wire 0 3, .wire 1 4),
    (.virt 5, .wire 1 5),
    (.wire 0 3, .wire 1 6),
    (.wire 1 7, .wire 0 8),
    (.virt 3, .wire 0 9),
    (.wire 0 7, .wire 0 10),
    (.wire 1 3, .virt 2),
    (.wire 0 11, .virt 2),
    (.wire 0 3, .virt 2)
  ]
  constants := [
    (.virt 2, 0),
    (.virt 3, 1)
  ]
  publicInputs := [
  ]

/-- Named target `x`. -/
def gadgetPinnedIntermediate.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetPinnedIntermediate.y : Target := .virt 1

/-- Named target `zero`. -/
def gadgetPinnedIntermediate.zero : Target := .virt 2

/-- Named target `diff`. -/
def gadgetPinnedIntermediate.diff : Target := .wire 0 3

/-- Named target `equal`. -/
def gadgetPinnedIntermediate.equal : Target := .virt 4

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetPinnedIntermediate.verifiers : List (String × List Target) :=
  []

theorem gadgetPinnedIntermediate_copies (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    a (.virt 0) = a (.wire 0 0) ∧ a (.virt 3) = a (.wire 0 1) ∧ a (.virt 1) = a (.wire 0 2) ∧ a (.virt 3) = a (.wire 0 4) ∧ a (.virt 3) = a (.wire 0 5) ∧ a (.virt 4) = a (.wire 0 6) ∧ a (.virt 4) = a (.wire 1 0) ∧ a (.wire 0 3) = a (.wire 1 1) ∧ a (.virt 4) = a (.wire 1 2) ∧ a (.wire 0 3) = a (.wire 1 4) ∧ a (.virt 5) = a (.wire 1 5) ∧ a (.wire 0 3) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 8) ∧ a (.virt 3) = a (.wire 0 9) ∧ a (.wire 0 7) = a (.wire 0 10) ∧ a (.wire 1 3) = a (.virt 2) ∧ a (.wire 0 11) = a (.virt 2) ∧ a (.wire 0 3) = a (.virt 2) := by
  have hc := h.2.1
  simp only [gadgetPinnedIntermediate, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetPinnedIntermediate_consts (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    a (.virt 2) = 0 ∧ a (.virt 3) = 1 := by
  have hconst := h.2.2
  simp only [gadgetPinnedIntermediate, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1⟩ := hconst
  exact ⟨k0, k1⟩

theorem gadgetPinnedIntermediate_f0 (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    a (.wire 0 3) = a (.virt 0) - a (.virt 1) := by
  have c0 := (gadgetPinnedIntermediate_copies a h).1
  have c1 := (gadgetPinnedIntermediate_copies a h).2.1
  have c2 := (gadgetPinnedIntermediate_copies a h).2.2.1
  obtain ⟨k0, k1⟩ := gadgetPinnedIntermediate_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, k1, ← c2] at e_0_0
  have hr := e_0_0
  linear_combination hr

theorem gadgetPinnedIntermediate_f1 (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 4)) (a (.virt 5)) := by
  have c0 := (gadgetPinnedIntermediate_copies a h).1
  have c1 := (gadgetPinnedIntermediate_copies a h).2.1
  have c2 := (gadgetPinnedIntermediate_copies a h).2.2.1
  have c3 := (gadgetPinnedIntermediate_copies a h).2.2.2.1
  have c4 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.1
  have c5 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.1
  have c6 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.1
  have c7 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.1
  have c8 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.1
  have c9 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1⟩ := gadgetPinnedIntermediate_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, k1, ← c2] at e_0_0
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_1
  simp only [← c3, k1, ← c4, k1, ← c5] at e_0_1
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_2
  simp only [← c12, ← c13, k1, ← c14] at e_0_2
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_0
  simp only [← c6, ← c7, ← c8] at e_1_0
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_1
  simp only [← c9, ← c10, ← c11] at e_1_1
  refine ⟨?_, ?_⟩
  · have hc := e_1_0
    simp only [e_0_0] at hc
    linear_combination c15.trans k0 - hc
  · have hc := e_0_2
    simp only [e_0_1, e_1_1, e_0_0] at hc
    linear_combination c16.trans k0 - hc

theorem gadgetPinnedIntermediate_f2 (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    a (.wire 0 3) = a (.virt 2) := by
  have c17 := (gadgetPinnedIntermediate_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c17

/-- Every satisfying assignment of `gadgetPinnedIntermediate` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetPinnedIntermediate_decode (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    a (.wire 0 3) = a (.virt 0) - a (.virt 1) ∧
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 4)) (a (.virt 5)) ∧
    a (.wire 0 3) = a (.virt 2) :=
  ⟨gadgetPinnedIntermediate_f0 a h, gadgetPinnedIntermediate_f1 a h, gadgetPinnedIntermediate_f2 a h⟩

/-- Constant folds that hold over `ℤ`: `nine = mul three three`, `eight = add three five`, `neg5 = sub zero five`. -/
def gadgetConstantFold (p : ℕ) : Circuit p where
  rows := [
  ]
  copies := [
  ]
  constants := [
    (.virt 0, 0),
    (.virt 4, 1),
    (.virt 1, 3),
    (.virt 2, 5),
    (.virt 5, 8),
    (.virt 3, 9),
    (.virt 6, (-5))
  ]
  publicInputs := [
  ]

/-- Named target `zero`. -/
def gadgetConstantFold.zero : Target := .virt 0

/-- Named target `three`. -/
def gadgetConstantFold.three : Target := .virt 1

/-- Named target `five`. -/
def gadgetConstantFold.five : Target := .virt 2

/-- Named target `nine`. -/
def gadgetConstantFold.nine : Target := .virt 3

/-- Named target `eight`. -/
def gadgetConstantFold.eight : Target := .virt 5

/-- Named target `neg5`. -/
def gadgetConstantFold.neg5 : Target := .virt 6

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetConstantFold.verifiers : List (String × List Target) :=
  []

theorem gadgetConstantFold_consts (a : Assignment p) (h : Satisfies (gadgetConstantFold p) a) :
    a (.virt 0) = 0 ∧ a (.virt 4) = 1 ∧ a (.virt 1) = 3 ∧ a (.virt 2) = 5 ∧ a (.virt 5) = 8 ∧ a (.virt 3) = 9 ∧ a (.virt 6) = -5 := by
  have hconst := h.2.2
  simp only [gadgetConstantFold, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3, k4, k5, k6⟩ := hconst
  exact ⟨k0, k1, k2, k3, k4, k5, k6⟩

theorem gadgetConstantFold_f0 (a : Assignment p) (h : Satisfies (gadgetConstantFold p) a) :
    a (.virt 3) = a (.virt 1) * a (.virt 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6⟩ := gadgetConstantFold_consts a h
  simp only [k2, k5]
  ring

theorem gadgetConstantFold_f1 (a : Assignment p) (h : Satisfies (gadgetConstantFold p) a) :
    a (.virt 5) = a (.virt 1) + a (.virt 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6⟩ := gadgetConstantFold_consts a h
  simp only [k2, k3, k4]
  ring

theorem gadgetConstantFold_f2 (a : Assignment p) (h : Satisfies (gadgetConstantFold p) a) :
    a (.virt 6) = a (.virt 0) - a (.virt 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6⟩ := gadgetConstantFold_consts a h
  simp only [k0, k3, k6]
  ring

/-- Every satisfying assignment of `gadgetConstantFold` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetConstantFold_decode (a : Assignment p) (h : Satisfies (gadgetConstantFold p) a) :
    a (.virt 3) = a (.virt 1) * a (.virt 1) ∧
    a (.virt 5) = a (.virt 1) + a (.virt 2) ∧
    a (.virt 6) = a (.virt 0) - a (.virt 2) :=
  ⟨gadgetConstantFold_f0 a h, gadgetConstantFold_f1 a h, gadgetConstantFold_f2 a h⟩

/-- One recorded call, `sum = add x y`. -/
def gadgetSingleFact (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, 1]⟩  -- row 0
  ]
  copies := [
    (.virt 0, .wire 0 0),
    (.virt 2, .wire 0 1),
    (.virt 1, .wire 0 2)
  ]
  constants := [
    (.virt 3, 0),
    (.virt 2, 1)
  ]
  publicInputs := [
  ]

/-- Named target `x`. -/
def gadgetSingleFact.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetSingleFact.y : Target := .virt 1

/-- Named target `sum`. -/
def gadgetSingleFact.sum : Target := .wire 0 3

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetSingleFact.verifiers : List (String × List Target) :=
  []

theorem gadgetSingleFact_copies (a : Assignment p) (h : Satisfies (gadgetSingleFact p) a) :
    a (.virt 0) = a (.wire 0 0) ∧ a (.virt 2) = a (.wire 0 1) ∧ a (.virt 1) = a (.wire 0 2) := by
  have hc := h.2.1
  simp only [gadgetSingleFact, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetSingleFact_consts (a : Assignment p) (h : Satisfies (gadgetSingleFact p) a) :
    a (.virt 3) = 0 ∧ a (.virt 2) = 1 := by
  have hconst := h.2.2
  simp only [gadgetSingleFact, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1⟩ := hconst
  exact ⟨k0, k1⟩

theorem gadgetSingleFact_f0 (a : Assignment p) (h : Satisfies (gadgetSingleFact p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
  have c0 := (gadgetSingleFact_copies a h).1
  have c1 := (gadgetSingleFact_copies a h).2.1
  have c2 := (gadgetSingleFact_copies a h).2.2
  obtain ⟨k0, k1⟩ := gadgetSingleFact_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, k1, ← c2] at e_0_0
  have hr := e_0_0
  linear_combination hr

/-- Every satisfying assignment of `gadgetSingleFact` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetSingleFact_decode (a : Assignment p) (h : Satisfies (gadgetSingleFact p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) :=
  gadgetSingleFact_f0 a h

/-- No recorded calls. -/
def gadgetNoFacts (p : ℕ) : Circuit p where
  rows := [
  ]
  copies := [
  ]
  constants := [
  ]
  publicInputs := [
  ]

/-- Named target `x`. -/
def gadgetNoFacts.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetNoFacts.y : Target := .virt 1

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetNoFacts.verifiers : List (String × List Target) :=
  []

/-- Every satisfying assignment of `gadgetNoFacts` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetNoFacts_decode (a : Assignment p) (h : Satisfies (gadgetNoFacts p) a) : True :=
  trivial

/-- `FACT_GROUP + 1` recorded calls (`connect x y` repeated): the last fact group of the decode theorem is a single fact. -/
def gadgetFactGroupBoundary (p : ℕ) : Circuit p where
  rows := [
  ]
  copies := [
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1),
    (.virt 0, .virt 1)
  ]
  constants := [
  ]
  publicInputs := [
  ]

/-- Named target `x`. -/
def gadgetFactGroupBoundary.x : Target := .virt 0

/-- Named target `y`. -/
def gadgetFactGroupBoundary.y : Target := .virt 1

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def gadgetFactGroupBoundary.verifiers : List (String × List Target) :=
  []

theorem gadgetFactGroupBoundary_copies (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) ∧ a (.virt 0) = a (.virt 1) := by
  have hc := h.2.1
  simp only [gadgetFactGroupBoundary, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem gadgetFactGroupBoundary_f0 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c0 := (gadgetFactGroupBoundary_copies a h).1
  exact c0

theorem gadgetFactGroupBoundary_f1 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c1 := (gadgetFactGroupBoundary_copies a h).2.1
  exact c1

theorem gadgetFactGroupBoundary_f2 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c2 := (gadgetFactGroupBoundary_copies a h).2.2.1
  exact c2

theorem gadgetFactGroupBoundary_f3 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c3 := (gadgetFactGroupBoundary_copies a h).2.2.2.1
  exact c3

theorem gadgetFactGroupBoundary_f4 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c4 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.1
  exact c4

theorem gadgetFactGroupBoundary_f5 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c5 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.1
  exact c5

theorem gadgetFactGroupBoundary_f6 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c6 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.1
  exact c6

theorem gadgetFactGroupBoundary_f7 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c7 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.1
  exact c7

theorem gadgetFactGroupBoundary_f8 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c8 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.1
  exact c8

theorem gadgetFactGroupBoundary_f9 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c9 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.1
  exact c9

theorem gadgetFactGroupBoundary_f10 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c10 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.1
  exact c10

theorem gadgetFactGroupBoundary_f11 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c11 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c11

theorem gadgetFactGroupBoundary_f12 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c12 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12

theorem gadgetFactGroupBoundary_f13 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c13 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13

theorem gadgetFactGroupBoundary_f14 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c14 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14

theorem gadgetFactGroupBoundary_f15 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c15 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15

theorem gadgetFactGroupBoundary_f16 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c16 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16

theorem gadgetFactGroupBoundary_f17 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c17 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17

theorem gadgetFactGroupBoundary_f18 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c18 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18

theorem gadgetFactGroupBoundary_f19 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c19 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19

theorem gadgetFactGroupBoundary_f20 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c20 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20

theorem gadgetFactGroupBoundary_f21 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c21 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21

theorem gadgetFactGroupBoundary_f22 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c22 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22

theorem gadgetFactGroupBoundary_f23 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c23 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c23

theorem gadgetFactGroupBoundary_f24 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c24 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c24

theorem gadgetFactGroupBoundary_f25 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c25 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c25

theorem gadgetFactGroupBoundary_f26 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c26 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c26

theorem gadgetFactGroupBoundary_f27 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c27 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c27

theorem gadgetFactGroupBoundary_f28 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c28 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c28

theorem gadgetFactGroupBoundary_f29 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c29 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c29

theorem gadgetFactGroupBoundary_f30 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c30 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c30

theorem gadgetFactGroupBoundary_f31 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c31 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c31

theorem gadgetFactGroupBoundary_f32 (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    a (.virt 0) = a (.virt 1) := by
  have c32 := (gadgetFactGroupBoundary_copies a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c32

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `gadgetFactGroupBoundary` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetFactGroupBoundary_decode (a : Assignment p) (h : Satisfies (gadgetFactGroupBoundary p) a) :
    (a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1) ∧
    a (.virt 0) = a (.virt 1)) ∧
    (a (.virt 0) = a (.virt 1)) :=
  ⟨⟨gadgetFactGroupBoundary_f0 a h, gadgetFactGroupBoundary_f1 a h, gadgetFactGroupBoundary_f2 a h, gadgetFactGroupBoundary_f3 a h, gadgetFactGroupBoundary_f4 a h, gadgetFactGroupBoundary_f5 a h, gadgetFactGroupBoundary_f6 a h, gadgetFactGroupBoundary_f7 a h, gadgetFactGroupBoundary_f8 a h, gadgetFactGroupBoundary_f9 a h, gadgetFactGroupBoundary_f10 a h, gadgetFactGroupBoundary_f11 a h, gadgetFactGroupBoundary_f12 a h, gadgetFactGroupBoundary_f13 a h, gadgetFactGroupBoundary_f14 a h, gadgetFactGroupBoundary_f15 a h, gadgetFactGroupBoundary_f16 a h, gadgetFactGroupBoundary_f17 a h, gadgetFactGroupBoundary_f18 a h, gadgetFactGroupBoundary_f19 a h, gadgetFactGroupBoundary_f20 a h, gadgetFactGroupBoundary_f21 a h, gadgetFactGroupBoundary_f22 a h, gadgetFactGroupBoundary_f23 a h, gadgetFactGroupBoundary_f24 a h, gadgetFactGroupBoundary_f25 a h, gadgetFactGroupBoundary_f26 a h, gadgetFactGroupBoundary_f27 a h, gadgetFactGroupBoundary_f28 a h, gadgetFactGroupBoundary_f29 a h, gadgetFactGroupBoundary_f30 a h, gadgetFactGroupBoundary_f31 a h⟩, gadgetFactGroupBoundary_f32 a h⟩

end Plonky2Spec.Generated
