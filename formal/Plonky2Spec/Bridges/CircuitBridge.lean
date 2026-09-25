/-
  Step 8 (spike) — machine-checked public-input decode from the **exported wiring**.

  `Generated/NullifierSelectCircuit.lean` is the pre-`build` constraint system the real
  `CircuitBuilder` lays down for the private-batch nullifier-selection path on 2 slots
  (`select(is_dummy, dnull_j, real_j)` per limb, outputs registered as public inputs),
  walked mechanically by the exporter. This module proves, from `Satisfies` alone:

  * `arithmeticRow20_iff` — the 20-op `ArithmeticGate` the standard config places is,
    constraint by constraint, the one-op `arithmeticGate_c0` on wires `4i..4i+3`; so the
    per-op reading `rowConstraints` gives `.arithmetic 20` is the real gate.
  * `nullifierSelect2_decode` — every satisfying assignment has boolean dummy flags and
    each public input equal to `bselect flag dnull real`. These are the decode facts
    `Plonky2Bridge` takes as hypotheses (`NullSlot.sel`, `nullifiers_val_bridge`), here
    *derived* from the copy constraints and constants the builder emitted.

  The proof is structurally mechanical (instantiate each op, orient each copy constraint
  as a rewrite, `ring`) but hand-indexed against the exported file; PLAN.md Step 8 records
  what it would take to generate it. `Satisfies` interprets only `ArithmeticGate` rows, so
  the theorem holds a fortiori for the real system.
-/
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.FinCases
import Plonky2Spec.Wiring
import Plonky2Spec.Boolean
import Plonky2Spec.Generated.NullifierSelectCircuit

namespace Plonky2Spec.Wiring

open Plonky2Spec.Generated

set_option linter.unusedSectionVars false

variable {p : ℕ} [Fact p.Prime]

/-- All 20 constraints of the real `ArithmeticGate { num_ops: 20 }` (the instance
    `CircuitConfig::standard_recursion_config()` places) on a row whose wires are `w`. -/
def arithmeticRow20 (w : ℕ → ZMod p) (c0 c1 : ZMod p) : Prop :=
  ∀ x ∈ [Generated.arithmeticGate20_c0, Generated.arithmeticGate20_c1, Generated.arithmeticGate20_c2, Generated.arithmeticGate20_c3, Generated.arithmeticGate20_c4,
      Generated.arithmeticGate20_c5, Generated.arithmeticGate20_c6, Generated.arithmeticGate20_c7, Generated.arithmeticGate20_c8, Generated.arithmeticGate20_c9,
      Generated.arithmeticGate20_c10, Generated.arithmeticGate20_c11, Generated.arithmeticGate20_c12, Generated.arithmeticGate20_c13, Generated.arithmeticGate20_c14,
      Generated.arithmeticGate20_c15, Generated.arithmeticGate20_c16, Generated.arithmeticGate20_c17, Generated.arithmeticGate20_c18, Generated.arithmeticGate20_c19].map
    (fun f => f (w 0) (w 1) (w 2) (w 3) (w 4) (w 5) (w 6) (w 7) (w 8) (w 9) (w 10) (w 11) (w 12) (w 13) (w 14) (w 15)
      (w 16) (w 17) (w 18) (w 19) (w 20) (w 21) (w 22) (w 23) (w 24) (w 25) (w 26) (w 27) (w 28) (w 29) (w 30) (w 31)
      (w 32) (w 33) (w 34) (w 35) (w 36) (w 37) (w 38) (w 39) (w 40) (w 41) (w 42) (w 43) (w 44) (w 45) (w 46) (w 47)
      (w 48) (w 49) (w 50) (w 51) (w 52) (w 53) (w 54) (w 55) (w 56) (w 57) (w 58) (w 59) (w 60) (w 61) (w 62) (w 63)
      (w 64) (w 65) (w 66) (w 67) (w 68) (w 69) (w 70) (w 71) (w 72) (w 73) (w 74) (w 75) (w 76) (w 77) (w 78) (w 79) c0 c1),
    x = 0

/-- The 20-op gate is op-by-op the one-op `arithmeticGate_c0` on wires `4i..4i+3`; so the
    per-op reading `rowConstraints` gives to `.arithmetic 20` is exactly the real gate. -/
theorem arithmeticRow20_iff (w : ℕ → ZMod p) (c0 c1 : ZMod p) :
    arithmeticRow20 w c0 c1 ↔
      ∀ i < 20, Generated.arithmeticGate_c0 (w (4 * i)) (w (4 * i + 1)) (w (4 * i + 2))
        (w (4 * i + 3)) c0 c1 = 0 := by
  unfold arithmeticRow20
  simp only [List.map_cons, List.map_nil, List.forall_mem_cons, List.not_mem_nil, false_implies,
    implies_true, and_true, Generated.arithmeticGate20_c0, Generated.arithmeticGate20_c1, Generated.arithmeticGate20_c2,
    Generated.arithmeticGate20_c3, Generated.arithmeticGate20_c4, Generated.arithmeticGate20_c5,
    Generated.arithmeticGate20_c6, Generated.arithmeticGate20_c7, Generated.arithmeticGate20_c8,
    Generated.arithmeticGate20_c9, Generated.arithmeticGate20_c10, Generated.arithmeticGate20_c11,
    Generated.arithmeticGate20_c12, Generated.arithmeticGate20_c13, Generated.arithmeticGate20_c14,
    Generated.arithmeticGate20_c15, Generated.arithmeticGate20_c16, Generated.arithmeticGate20_c17,
    Generated.arithmeticGate20_c18, Generated.arithmeticGate20_c19,
    Generated.arithmeticGate_c0]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ i hi
    interval_cases i <;> simpa
  · intro h
    exact ⟨by simpa using h 0 (by norm_num), by simpa using h 1 (by norm_num),
      by simpa using h 2 (by norm_num), by simpa using h 3 (by norm_num),
      by simpa using h 4 (by norm_num), by simpa using h 5 (by norm_num),
      by simpa using h 6 (by norm_num), by simpa using h 7 (by norm_num),
      by simpa using h 8 (by norm_num), by simpa using h 9 (by norm_num),
      by simpa using h 10 (by norm_num), by simpa using h 11 (by norm_num),
      by simpa using h 12 (by norm_num), by simpa using h 13 (by norm_num),
      by simpa using h 14 (by norm_num), by simpa using h 15 (by norm_num),
      by simpa using h 16 (by norm_num), by simpa using h 17 (by norm_num),
      by simpa using h 18 (by norm_num), by simpa using h 19 (by norm_num)⟩


/-- `rowConstraints` on an `.arithmetic 20` row *is* the real 20-op gate on that row. -/
theorem rowConstraints_arithmetic20 (a : Assignment p) (row : ℕ) (c0 c1 : ZMod p) :
    rowConstraints a row ⟨.arithmetic 20, [c0, c1]⟩ ↔
      arithmeticRow20 (fun k => a (.wire row k)) c0 c1 := by
  simp only [rowConstraints, arithOp, List.getD_cons_zero, List.getD_cons_succ]
  exact (arithmeticRow20_iff (p := p) (fun k => a (.wire row k)) c0 c1).symm

/-- Slot owning public input `k` of `nullifierSelect2` (`k / 4`). -/
def slotOf : Fin 8 → Fin 2 := ![0, 0, 0, 0, 1, 1, 1, 1]

/-- The exported public-input list is the named `out` vector. -/
theorem nullifierSelect2_publicInputs :
    (nullifierSelect2 p).publicInputs = List.ofFn nullifierSelect2.out := by
  rfl

theorem nullifierSelect2_decode (a : Assignment p) (h : Satisfies (nullifierSelect2 p) a) :
    (∀ s : Fin 2, IsBool (a (nullifierSelect2.isDummy s))) ∧
    ∀ k : Fin 8, a (nullifierSelect2.out k) =
      bselect (a (nullifierSelect2.isDummy (slotOf k))) (a (nullifierSelect2.dnull k))
        (a (nullifierSelect2.real k)) := by
  obtain ⟨hrows, hcopy, hconst⟩ := h
  simp only [nullifierSelect2, rowsSatisfied, rowConstraints, List.getD_cons_zero,
    List.getD_cons_succ, and_true] at hrows
  simp only [nullifierSelect2, List.forall_mem_cons] at hcopy hconst
  obtain ⟨hzero, -⟩ := hconst
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13,
    c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27,
    c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41,
    c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55⟩ := hcopy

  have e0 := (arithOp_iff a 0 0 _ _).mp (hrows 0 (by norm_num))
  have e1 := (arithOp_iff a 0 1 _ _).mp (hrows 1 (by norm_num))
  have e2 := (arithOp_iff a 0 2 _ _).mp (hrows 2 (by norm_num))
  have e3 := (arithOp_iff a 0 3 _ _).mp (hrows 3 (by norm_num))
  have e4 := (arithOp_iff a 0 4 _ _).mp (hrows 4 (by norm_num))
  have e5 := (arithOp_iff a 0 5 _ _).mp (hrows 5 (by norm_num))
  have e6 := (arithOp_iff a 0 6 _ _).mp (hrows 6 (by norm_num))
  have e7 := (arithOp_iff a 0 7 _ _).mp (hrows 7 (by norm_num))
  have e8 := (arithOp_iff a 0 8 _ _).mp (hrows 8 (by norm_num))
  have e9 := (arithOp_iff a 0 9 _ _).mp (hrows 9 (by norm_num))
  have e10 := (arithOp_iff a 0 10 _ _).mp (hrows 10 (by norm_num))
  have e11 := (arithOp_iff a 0 11 _ _).mp (hrows 11 (by norm_num))
  have e12 := (arithOp_iff a 0 12 _ _).mp (hrows 12 (by norm_num))
  have e13 := (arithOp_iff a 0 13 _ _).mp (hrows 13 (by norm_num))
  have e14 := (arithOp_iff a 0 14 _ _).mp (hrows 14 (by norm_num))
  have e15 := (arithOp_iff a 0 15 _ _).mp (hrows 15 (by norm_num))
  have e16 := (arithOp_iff a 0 16 _ _).mp (hrows 16 (by norm_num))
  have e17 := (arithOp_iff a 0 17 _ _).mp (hrows 17 (by norm_num))
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17
  simp only [c3, c31, ← c0, ← c1, ← c2, ← c4, ← c5, ← c6, ← c7, ← c8,
    ← c9, ← c10, ← c11, ← c12, ← c13, ← c14, ← c15, ← c16, ← c17, ← c18,
    ← c19, ← c20, ← c21, ← c22, ← c23, ← c24, ← c25, ← c26, ← c27, ← c28,
    ← c29, ← c30, ← c32, ← c33, ← c34, ← c35, ← c36, ← c37, ← c38, ← c39,
    ← c40, ← c41, ← c42, ← c43, ← c44, ← c45, ← c46, ← c47, ← c48, ← c49,
    ← c50, ← c51, ← c52, ← c53, ← c54, ← c55] at e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17
  refine ⟨?_, ?_⟩
  · simp only [Fin.forall_fin_two, isBool_iff_assertBool, nullifierSelect2.isDummy,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    constructor
    · linear_combination hzero - e0
    · linear_combination hzero - e9
  · simp only [Fin.forall_fin_succ, Fin.forall_fin_zero, nullifierSelect2.out,
      nullifierSelect2.isDummy, nullifierSelect2.dnull, nullifierSelect2.real, slotOf, bselect,
      Matrix.cons_val_zero, Matrix.cons_val_succ, Matrix.cons_val_one, and_true]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [e2, e1]; ring
    · rw [e4, e3]; ring
    · rw [e6, e5]; ring
    · rw [e8, e7]; ring
    · rw [e11, e10]; ring
    · rw [e13, e12]; ring
    · rw [e15, e14]; ring
    · rw [e17, e16]; ring

end Plonky2Spec.Wiring
