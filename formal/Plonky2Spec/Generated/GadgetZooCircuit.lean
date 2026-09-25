/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) by building the
  gadget-zoo circuit through the recording builder and walking its pre-`build`
  constraint system. The theorem's proof is generated too, one block per recorded
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

/-- The gadget zoo: `assert_bool flag`, `eq = is_equal x y`, `sel = select flag x y`, `either = or eq flag`, `nflag = not flag`, `both = and eq nflag`, `head = sub 10000 fee`, `range_check head 14`, `connect sel either`; public inputs `x`, `sel`, `both`. -/
def gadgetZoo (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 20, [(-1), 1]⟩,  -- row 2
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 3
    ⟨.baseSum2 63, []⟩  -- row 4
  ]
  copies := [
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
    (.virt 6, .wire 3 1),
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
    (.wire 4 36, .virt 4),
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

variable {p : ℕ} [Fact p.Prime]

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
    a (.wire 0 23) = a (.wire 3 3) := by
  have hcopy := h.2.1
  have hconst := h.2.2
  simp only [gadgetZoo, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy hconst
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92⟩ := hcopy
  obtain ⟨k0, k1, k2, k3⟩ := hconst
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by norm_num)
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by norm_num)
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by norm_num)
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by norm_num)
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by norm_num)
  have e_0_3 := arithEq_of_rows h (row := 0) (i := 3) rfl (by norm_num)
  have e_0_4 := arithEq_of_rows h (row := 0) (i := 4) rfl (by norm_num)
  have e_0_5 := arithEq_of_rows h (row := 0) (i := 5) rfl (by norm_num)
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by norm_num)
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by norm_num)
  have e_0_6 := arithEq_of_rows h (row := 0) (i := 6) rfl (by norm_num)
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by norm_num)
  have e_0_7 := arithEq_of_rows h (row := 0) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_0 e_0_1 e_0_2 e_1_0 e_1_1 e_0_3 e_0_4 e_0_5 e_2_0 e_3_0 e_0_6 e_1_2 e_0_7
  simp only [← c0, ← c1, ← c2, ← c4, ← c5, ← c6, ← c7, ← c8, ← c9, ← c10, ← c11, ← c12, ← c13, ← c14, ← c15, ← c16, ← c17, ← c18, ← c21, ← c22, ← c23, ← c24, ← c25, ← c26, ← c27, ← c28, ← c29, ← c30, ← c31, ← c32, ← c33, ← c34, ← c35, ← c36, ← c37, ← c38, ← c39, ← c40, ← c41, c3, c19, c20, k0, k1, k2, k3] at e_0_0 e_0_1 e_0_2 e_1_0 e_1_1 e_0_3 e_0_4 e_0_5 e_2_0 e_3_0 e_0_6 e_1_2 e_0_7
  have f0 : IsBool (a (.virt 3)) := by
    exact isBool_iff_assertBool.mpr (by linear_combination -e_0_0)
  have f1 : IsEqual (a (.virt 0)) (a (.virt 1)) (a (.virt 5)) (a (.virt 7)) := by
    simp only [e_0_1, e_1_1, e_0_2, k0, k1, k2, k3] at e_1_0 e_0_3
    exact ⟨by linear_combination -e_1_0, by linear_combination -e_0_3⟩
  have f2 : a (.wire 0 23) = bselect (a (.virt 3)) (a (.virt 0)) (a (.virt 1)) := by
    simp only [bselect, e_0_5, e_0_4, k0, k1, k2, k3]
    ring
  have f3 : a (.wire 3 3) = bor (a (.virt 5)) (a (.virt 3)) := by
    simp only [bor, e_3_0, e_2_0, k0, k1, k2, k3]
    ring
  have f4 : a (.wire 0 27) = bnot (a (.virt 3)) := by
    simp only [bnot, e_0_6, k0, k1, k2, k3]
    ring
  have f5 : a (.wire 1 11) = band (a (.virt 5)) (a (.wire 0 27)) := by
    simp only [band, e_1_2, k0, k1, k2, k3]
    ring
  have f6 : a (.wire 0 31) = a (.virt 8) - a (.virt 2) := by
    simp only [e_0_7, k0, k1, k2, k3]
    ring
  have f7 : rangeCheck (a (.wire 0 31)) 14 := by
    have hr := rangeCheck_of_row h (row := 4) (N := 63) (n := 14) rfl rfl (by norm_num) (by
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
  have f8 : a (.wire 0 23) = a (.wire 3 3) := by
    exact c92
  exact ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8⟩

end Plonky2Spec.Generated
