/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) by building the
  gadget edge-case circuit(s) through the recording builder and walking the pre-`build`
  constraint system. Each theorem's proof is generated too, one block per recorded
  gadget call, from the ops and copy constraints the builder emitted for it.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

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
    a (.wire 2 11) = a (.virt 2) := by
  have hcopy := h.2.1
  have hconst := h.2.2
  simp only [gadgetEdgeCases, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy hconst
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54⟩ := hcopy
  obtain ⟨k0, k1⟩ := hconst
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by norm_num)
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by norm_num)
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by norm_num)
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by norm_num)
  have e_1_3 := arithEq_of_rows h (row := 1) (i := 3) rfl (by norm_num)
  have e_1_4 := arithEq_of_rows h (row := 1) (i := 4) rfl (by norm_num)
  have e_1_5 := arithEq_of_rows h (row := 1) (i := 5) rfl (by norm_num)
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by norm_num)
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by norm_num)
  have e_1_6 := arithEq_of_rows h (row := 1) (i := 6) rfl (by norm_num)
  have e_1_7 := arithEq_of_rows h (row := 1) (i := 7) rfl (by norm_num)
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by norm_num)
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by norm_num)
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by norm_num)
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_0 e_1_0 e_1_1 e_1_2 e_1_3 e_1_4 e_1_5 e_2_0 e_2_1 e_1_6 e_1_7 e_1_8 e_2_2 e_2_3 e_1_9
  simp only [← c0, ← c1, ← c2, ← c4, ← c5, ← c6, ← c7, ← c8, ← c9, ← c12, ← c13, ← c14, ← c15, ← c16, ← c17, ← c20, ← c21, ← c22, ← c23, ← c24, ← c25, ← c26, ← c27, ← c28, ← c29, ← c30, ← c31, ← c32, ← c33, ← c34, ← c37, ← c38, ← c39, ← c40, ← c41, ← c42, ← c43, ← c44, ← c45, ← c46, ← c47, ← c48, ← c49, ← c50, ← c51, k0, k1] at e_0_0 e_1_0 e_1_1 e_1_2 e_1_3 e_1_4 e_1_5 e_2_0 e_2_1 e_1_6 e_1_7 e_1_8 e_2_2 e_2_3 e_1_9
  have f0 : a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
    have hr := e_0_0
    linear_combination hr
  have f1 : a (.wire 0 3) = a (.virt 2) := by
    exact c3
  have f2 : IsEqual (a (.virt 2)) (a (.virt 2)) (a (.virt 4)) (a (.virt 5)) := by
    simp only [k0]
    refine ⟨?_, ?_⟩
    · ring
    · have hc := e_1_1
      simp only [e_1_0] at hc
      linear_combination c11.trans k0 - hc
  have f3 : IsEqual (a (.virt 3)) (a (.virt 2)) (a (.virt 6)) (a (.virt 7)) := by
    simp only [k0, k1]
    refine ⟨?_, ?_⟩
    · rw [c18.trans k0]
      ring
    · have hc := e_1_3
      simp only [e_1_2] at hc
      linear_combination c19.trans k0 - hc
  have f4 : IsEqual (a (.virt 0)) (a (.virt 0)) (a (.virt 8)) (a (.virt 9)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_0
      simp only [e_1_5] at hc
      linear_combination c35.trans k0 - hc
    · have hc := e_1_6
      simp only [e_1_4, e_2_1, e_1_5] at hc
      linear_combination c36.trans k0 - hc
  have f5 : IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 10)) (a (.virt 11)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_2
      simp only [e_1_8] at hc
      linear_combination c52.trans k0 - hc
    · have hc := e_1_9
      simp only [e_1_7, e_2_3, e_1_8] at hc
      linear_combination c53.trans k0 - hc
  have f6 : a (.wire 1 35) = a (.virt 0) - a (.virt 1) := by
    have hr := e_1_8
    linear_combination hr
  have f7 : a (.wire 2 11) = a (.virt 10) * a (.wire 1 35) := by
    have hr := e_2_2
    linear_combination hr
  have f8 : a (.wire 2 11) = a (.virt 2) := by
    exact c54
  exact ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8⟩

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

/-- Every satisfying assignment of `gadgetIdentityFold` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetIdentityFold_decode (a : Assignment p) (h : Satisfies (gadgetIdentityFold p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) ∧
    a (.wire 0 3) = a (.wire 0 3) * a (.virt 2) := by
  have hcopy := h.2.1
  have hconst := h.2.2
  simp only [gadgetIdentityFold, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy hconst
  obtain ⟨c0, c1, c2⟩ := hcopy
  obtain ⟨k0, k1⟩ := hconst
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, ← c2, k0, k1] at e_0_0
  have f0 : a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
    have hr := e_0_0
    linear_combination hr
  have f1 : a (.wire 0 3) = a (.wire 0 3) * a (.virt 2) := by
    simp only [k1]
    ring
  exact ⟨f0, f1⟩

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

/-- Every satisfying assignment of `gadgetPinnedIntermediate` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetPinnedIntermediate_decode (a : Assignment p) (h : Satisfies (gadgetPinnedIntermediate p) a) :
    a (.wire 0 3) = a (.virt 0) - a (.virt 1) ∧
    IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 4)) (a (.virt 5)) ∧
    a (.wire 0 3) = a (.virt 2) := by
  have hcopy := h.2.1
  have hconst := h.2.2
  simp only [gadgetPinnedIntermediate, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy hconst
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17⟩ := hcopy
  obtain ⟨k0, k1⟩ := hconst
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by norm_num)
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by norm_num)
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by norm_num)
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by norm_num)
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_0 e_0_1 e_1_0 e_1_1 e_0_2
  simp only [← c0, ← c1, ← c2, ← c3, ← c4, ← c5, ← c6, ← c7, ← c8, ← c9, ← c10, ← c11, ← c12, ← c13, ← c14, k0, k1] at e_0_0 e_0_1 e_1_0 e_1_1 e_0_2
  have f0 : a (.wire 0 3) = a (.virt 0) - a (.virt 1) := by
    have hr := e_0_0
    linear_combination hr
  have f1 : IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 4)) (a (.virt 5)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_1_0
      simp only [e_0_0] at hc
      linear_combination c15.trans k0 - hc
    · have hc := e_0_2
      simp only [e_0_1, e_1_1, e_0_0] at hc
      linear_combination c16.trans k0 - hc
  have f2 : a (.wire 0 3) = a (.virt 2) := by
    exact c17
  exact ⟨f0, f1, f2⟩

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

/-- Every satisfying assignment of `gadgetSingleFact` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetSingleFact_decode (a : Assignment p) (h : Satisfies (gadgetSingleFact p) a) :
    a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
  have hcopy := h.2.1
  have hconst := h.2.2
  simp only [gadgetSingleFact, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy hconst
  obtain ⟨c0, c1, c2⟩ := hcopy
  obtain ⟨k0, k1⟩ := hconst
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, ← c2, k0, k1] at e_0_0
  have f0 : a (.wire 0 3) = a (.virt 0) + a (.virt 1) := by
    have hr := e_0_0
    linear_combination hr
  exact f0

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

/-- Every satisfying assignment of `gadgetNoFacts` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem gadgetNoFacts_decode (a : Assignment p) (h : Satisfies (gadgetNoFacts p) a) : True :=
  trivial

end Plonky2Spec.Generated
