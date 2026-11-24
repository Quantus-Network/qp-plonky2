/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PrivateBatchWrapper8`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge
import Plonky2Spec.Generated.PrivateBatchWrapper8.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem privateBatchWrapper8_f0 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 5)) (a (.virt 11)) (a (.virt 284)) (a (.virt 285)) := by
  have c0 := (privateBatchWrapper8_copies0 a h).1
  have c1 := (privateBatchWrapper8_copies0 a h).2.1
  have c2 := (privateBatchWrapper8_copies0 a h).2.2.1
  have c3 := (privateBatchWrapper8_copies0 a h).2.2.2.1
  have c4 := (privateBatchWrapper8_copies0 a h).2.2.2.2.1
  have c5 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.1
  have c6 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.1
  have c7 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c3, ← c4, ← c5] at e_1_12
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c6, ← c7, ← c8] at e_1_13
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c0, k3, ← c1, k3, ← c2] at e_2_0
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c9, ← c10, k3, ← c11] at e_2_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_12
    linear_combination c12.trans k2 - hc
  · have hc := e_2_1
    simp only [e_2_0, e_1_13] at hc
    linear_combination c13.trans k2 - hc

theorem privateBatchWrapper8_f1 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 6)) (a (.virt 11)) (a (.virt 286)) (a (.virt 287)) := by
  have c14 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c17, ← c18, ← c19] at e_1_14
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c14, k3, ← c15, k3, ← c16] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c23, ← c24, k3, ← c25] at e_2_3
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c20, ← c21, ← c22] at e_3_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_14
    linear_combination c26.trans k2 - hc
  · have hc := e_2_3
    simp only [e_2_2, e_3_0] at hc
    linear_combination c27.trans k2 - hc

theorem privateBatchWrapper8_f2 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 7)) (a (.virt 11)) (a (.virt 288)) (a (.virt 289)) := by
  have c28 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (privateBatchWrapper8_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (privateBatchWrapper8_copies1 a h).1
  have c33 := (privateBatchWrapper8_copies1 a h).2.1
  have c34 := (privateBatchWrapper8_copies1 a h).2.2.1
  have c35 := (privateBatchWrapper8_copies1 a h).2.2.2.1
  have c36 := (privateBatchWrapper8_copies1 a h).2.2.2.2.1
  have c37 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.1
  have c38 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.1
  have c39 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c28, k3, ← c29, k3, ← c30] at e_2_4
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c37, ← c38, k3, ← c39] at e_2_5
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c31, ← c32, ← c33] at e_3_1
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c34, ← c35, ← c36] at e_3_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_1
    linear_combination c40.trans k2 - hc
  · have hc := e_2_5
    simp only [e_2_4, e_3_2] at hc
    linear_combination c41.trans k2 - hc

theorem privateBatchWrapper8_f3 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 8)) (a (.virt 11)) (a (.virt 290)) (a (.virt 291)) := by
  have c42 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c42, k3, ← c43, k3, ← c44] at e_2_6
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c51, ← c52, k3, ← c53] at e_2_7
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c45, ← c46, ← c47] at e_3_3
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c48, ← c49, ← c50] at e_3_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_3
    linear_combination c54.trans k2 - hc
  · have hc := e_2_7
    simp only [e_2_6, e_3_4] at hc
    linear_combination c55.trans k2 - hc

theorem privateBatchWrapper8_f4 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 3 23) = band (a (.virt 284)) (a (.virt 286)) := by
  have c56 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c56, ← c57, ← c58] at e_3_5
  have hr := e_3_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f5 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 3 27) = band (a (.virt 288)) (a (.virt 290)) := by
  have c59 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c59, ← c60, ← c61] at e_3_6
  have hr := e_3_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f6 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 3 31) = band (a (.wire 3 23)) (a (.wire 3 27)) := by
  have c62 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (privateBatchWrapper8_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (privateBatchWrapper8_copies2 a h).1
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c62, ← c63, ← c64] at e_3_7
  have hr := e_3_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f7 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 18)) (a (.virt 11)) (a (.virt 292)) (a (.virt 293)) := by
  have c65 := (privateBatchWrapper8_copies2 a h).2.1
  have c66 := (privateBatchWrapper8_copies2 a h).2.2.1
  have c67 := (privateBatchWrapper8_copies2 a h).2.2.2.1
  have c68 := (privateBatchWrapper8_copies2 a h).2.2.2.2.1
  have c69 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.1
  have c70 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.1
  have c71 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c65, k3, ← c66, k3, ← c67] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c74, ← c75, k3, ← c76] at e_2_9
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c68, ← c69, ← c70] at e_3_8
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c71, ← c72, ← c73] at e_3_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_8
    linear_combination c77.trans k2 - hc
  · have hc := e_2_9
    simp only [e_2_8, e_3_9] at hc
    linear_combination c78.trans k2 - hc

theorem privateBatchWrapper8_f8 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 19)) (a (.virt 11)) (a (.virt 294)) (a (.virt 295)) := by
  have c79 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c79, k3, ← c80, k3, ← c81] at e_2_10
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c88, ← c89, k3, ← c90] at e_2_11
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c82, ← c83, ← c84] at e_3_10
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c85, ← c86, ← c87] at e_3_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_10
    linear_combination c91.trans k2 - hc
  · have hc := e_2_11
    simp only [e_2_10, e_3_11] at hc
    linear_combination c92.trans k2 - hc

theorem privateBatchWrapper8_f9 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 20)) (a (.virt 11)) (a (.virt 296)) (a (.virt 297)) := by
  have c93 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (privateBatchWrapper8_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (privateBatchWrapper8_copies3 a h).1
  have c97 := (privateBatchWrapper8_copies3 a h).2.1
  have c98 := (privateBatchWrapper8_copies3 a h).2.2.1
  have c99 := (privateBatchWrapper8_copies3 a h).2.2.2.1
  have c100 := (privateBatchWrapper8_copies3 a h).2.2.2.2.1
  have c101 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.1
  have c102 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.1
  have c103 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c93, k3, ← c94, k3, ← c95] at e_2_12
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c102, ← c103, k3, ← c104] at e_2_13
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c96, ← c97, ← c98] at e_3_12
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c99, ← c100, ← c101] at e_3_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_12
    linear_combination c105.trans k2 - hc
  · have hc := e_2_13
    simp only [e_2_12, e_3_13] at hc
    linear_combination c106.trans k2 - hc

theorem privateBatchWrapper8_f10 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 21)) (a (.virt 11)) (a (.virt 298)) (a (.virt 299)) := by
  have c107 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c107, k3, ← c108, k3, ← c109] at e_2_14
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c110, ← c111, ← c112] at e_3_14
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c113, ← c114, ← c115] at e_4_0
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c116, ← c117, k3, ← c118] at e_5_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_14
    linear_combination c119.trans k2 - hc
  · have hc := e_5_0
    simp only [e_2_14, e_4_0] at hc
    linear_combination c120.trans k2 - hc

theorem privateBatchWrapper8_f11 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 4 7) = band (a (.virt 292)) (a (.virt 294)) := by
  have c121 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c121, ← c122, ← c123] at e_4_1
  have hr := e_4_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f12 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 4 11) = band (a (.virt 296)) (a (.virt 298)) := by
  have c124 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c124, ← c125, ← c126] at e_4_2
  have hr := e_4_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f13 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11)) := by
  have c127 := (privateBatchWrapper8_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (privateBatchWrapper8_copies4 a h).1
  have c129 := (privateBatchWrapper8_copies4 a h).2.1
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c127, ← c128, ← c129] at e_4_3
  have hr := e_4_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f14 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 29)) (a (.virt 11)) (a (.virt 300)) (a (.virt 301)) := by
  have c130 := (privateBatchWrapper8_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper8_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper8_copies4 a h).2.2.2.2.1
  have c133 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.1
  have c134 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.1
  have c135 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c133, ← c134, ← c135] at e_4_4
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c136, ← c137, ← c138] at e_4_5
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c130, k3, ← c131, k3, ← c132] at e_5_1
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c139, ← c140, k3, ← c141] at e_5_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_4
    linear_combination c142.trans k2 - hc
  · have hc := e_5_2
    simp only [e_5_1, e_4_5] at hc
    linear_combination c143.trans k2 - hc

theorem privateBatchWrapper8_f15 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 30)) (a (.virt 11)) (a (.virt 302)) (a (.virt 303)) := by
  have c144 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c145 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c148 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c151 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c147, ← c148, ← c149] at e_4_6
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c150, ← c151, ← c152] at e_4_7
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c144, k3, ← c145, k3, ← c146] at e_5_3
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c153, ← c154, k3, ← c155] at e_5_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_6
    linear_combination c156.trans k2 - hc
  · have hc := e_5_4
    simp only [e_5_3, e_4_7] at hc
    linear_combination c157.trans k2 - hc

theorem privateBatchWrapper8_f16 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 31)) (a (.virt 11)) (a (.virt 304)) (a (.virt 305)) := by
  have c158 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (privateBatchWrapper8_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (privateBatchWrapper8_copies5 a h).1
  have c161 := (privateBatchWrapper8_copies5 a h).2.1
  have c162 := (privateBatchWrapper8_copies5 a h).2.2.1
  have c163 := (privateBatchWrapper8_copies5 a h).2.2.2.1
  have c164 := (privateBatchWrapper8_copies5 a h).2.2.2.2.1
  have c165 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.1
  have c166 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.1
  have c167 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c161, ← c162, ← c163] at e_4_8
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c164, ← c165, ← c166] at e_4_9
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c158, k3, ← c159, k3, ← c160] at e_5_5
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c167, ← c168, k3, ← c169] at e_5_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_8
    linear_combination c170.trans k2 - hc
  · have hc := e_5_6
    simp only [e_5_5, e_4_9] at hc
    linear_combination c171.trans k2 - hc

theorem privateBatchWrapper8_f17 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 32)) (a (.virt 11)) (a (.virt 306)) (a (.virt 307)) := by
  have c172 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c175, ← c176, ← c177] at e_4_10
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c178, ← c179, ← c180] at e_4_11
  have e_5_7 := arithEq_of_rows h (row := 5) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c172, k3, ← c173, k3, ← c174] at e_5_7
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c181, ← c182, k3, ← c183] at e_5_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_10
    linear_combination c184.trans k2 - hc
  · have hc := e_5_8
    simp only [e_5_7, e_4_11] at hc
    linear_combination c185.trans k2 - hc

theorem privateBatchWrapper8_f18 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 4 51) = band (a (.virt 300)) (a (.virt 302)) := by
  have c186 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c186, ← c187, ← c188] at e_4_12
  have hr := e_4_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f19 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 4 55) = band (a (.virt 304)) (a (.virt 306)) := by
  have c189 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (privateBatchWrapper8_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c189, ← c190, ← c191] at e_4_13
  have hr := e_4_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f20 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 4 59) = band (a (.wire 4 51)) (a (.wire 4 55)) := by
  have c192 := (privateBatchWrapper8_copies6 a h).1
  have c193 := (privateBatchWrapper8_copies6 a h).2.1
  have c194 := (privateBatchWrapper8_copies6 a h).2.2.1
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c192, ← c193, ← c194] at e_4_14
  have hr := e_4_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f21 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 40)) (a (.virt 11)) (a (.virt 308)) (a (.virt 309)) := by
  have c195 := (privateBatchWrapper8_copies6 a h).2.2.2.1
  have c196 := (privateBatchWrapper8_copies6 a h).2.2.2.2.1
  have c197 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.1
  have c198 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.1
  have c199 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_9 := arithEq_of_rows h (row := 5) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c195, k3, ← c196, k3, ← c197] at e_5_9
  have e_5_10 := arithEq_of_rows h (row := 5) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c204, ← c205, k3, ← c206] at e_5_10
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c198, ← c199, ← c200] at e_6_0
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c201, ← c202, ← c203] at e_6_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_0
    linear_combination c207.trans k2 - hc
  · have hc := e_5_10
    simp only [e_5_9, e_6_1] at hc
    linear_combination c208.trans k2 - hc

theorem privateBatchWrapper8_f22 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 41)) (a (.virt 11)) (a (.virt 310)) (a (.virt 311)) := by
  have c209 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_11 := arithEq_of_rows h (row := 5) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c209, k3, ← c210, k3, ← c211] at e_5_11
  have e_5_12 := arithEq_of_rows h (row := 5) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c218, ← c219, k3, ← c220] at e_5_12
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c212, ← c213, ← c214] at e_6_2
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c215, ← c216, ← c217] at e_6_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_2
    linear_combination c221.trans k2 - hc
  · have hc := e_5_12
    simp only [e_5_11, e_6_3] at hc
    linear_combination c222.trans k2 - hc

theorem privateBatchWrapper8_f23 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 42)) (a (.virt 11)) (a (.virt 312)) (a (.virt 313)) := by
  have c223 := (privateBatchWrapper8_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (privateBatchWrapper8_copies7 a h).1
  have c225 := (privateBatchWrapper8_copies7 a h).2.1
  have c226 := (privateBatchWrapper8_copies7 a h).2.2.1
  have c227 := (privateBatchWrapper8_copies7 a h).2.2.2.1
  have c228 := (privateBatchWrapper8_copies7 a h).2.2.2.2.1
  have c229 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.1
  have c230 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.1
  have c231 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_13 := arithEq_of_rows h (row := 5) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c223, k3, ← c224, k3, ← c225] at e_5_13
  have e_5_14 := arithEq_of_rows h (row := 5) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c232, ← c233, k3, ← c234] at e_5_14
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c226, ← c227, ← c228] at e_6_4
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c229, ← c230, ← c231] at e_6_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_4
    linear_combination c235.trans k2 - hc
  · have hc := e_5_14
    simp only [e_5_13, e_6_5] at hc
    linear_combination c236.trans k2 - hc

theorem privateBatchWrapper8_f24 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 43)) (a (.virt 11)) (a (.virt 314)) (a (.virt 315)) := by
  have c237 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c240, ← c241, ← c242] at e_6_6
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c243, ← c244, ← c245] at e_6_7
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c237, k3, ← c238, k3, ← c239] at e_7_0
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c246, ← c247, k3, ← c248] at e_7_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_6
    linear_combination c249.trans k2 - hc
  · have hc := e_7_1
    simp only [e_7_0, e_6_7] at hc
    linear_combination c250.trans k2 - hc

theorem privateBatchWrapper8_f25 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 6 35) = band (a (.virt 308)) (a (.virt 310)) := by
  have c251 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c251, ← c252, ← c253] at e_6_8
  have hr := e_6_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f26 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 6 39) = band (a (.virt 312)) (a (.virt 314)) := by
  have c254 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (privateBatchWrapper8_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (privateBatchWrapper8_copies8 a h).1
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c254, ← c255, ← c256] at e_6_9
  have hr := e_6_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f27 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 6 43) = band (a (.wire 6 35)) (a (.wire 6 39)) := by
  have c257 := (privateBatchWrapper8_copies8 a h).2.1
  have c258 := (privateBatchWrapper8_copies8 a h).2.2.1
  have c259 := (privateBatchWrapper8_copies8 a h).2.2.2.1
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c257, ← c258, ← c259] at e_6_10
  have hr := e_6_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f28 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 51)) (a (.virt 11)) (a (.virt 316)) (a (.virt 317)) := by
  have c260 := (privateBatchWrapper8_copies8 a h).2.2.2.2.1
  have c261 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.1
  have c263 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c263, ← c264, ← c265] at e_6_11
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c266, ← c267, ← c268] at e_6_12
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c260, k3, ← c261, k3, ← c262] at e_7_2
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c269, ← c270, k3, ← c271] at e_7_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_11
    linear_combination c272.trans k2 - hc
  · have hc := e_7_3
    simp only [e_7_2, e_6_12] at hc
    linear_combination c273.trans k2 - hc

theorem privateBatchWrapper8_f29 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 52)) (a (.virt 11)) (a (.virt 318)) (a (.virt 319)) := by
  have c274 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c278 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (privateBatchWrapper8_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c277, ← c278, ← c279] at e_6_13
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c280, ← c281, ← c282] at e_6_14
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c274, k3, ← c275, k3, ← c276] at e_7_4
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c283, ← c284, k3, ← c285] at e_7_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_13
    linear_combination c286.trans k2 - hc
  · have hc := e_7_5
    simp only [e_7_4, e_6_14] at hc
    linear_combination c287.trans k2 - hc

theorem privateBatchWrapper8_f30 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 53)) (a (.virt 11)) (a (.virt 320)) (a (.virt 321)) := by
  have c288 := (privateBatchWrapper8_copies9 a h).1
  have c289 := (privateBatchWrapper8_copies9 a h).2.1
  have c290 := (privateBatchWrapper8_copies9 a h).2.2.1
  have c291 := (privateBatchWrapper8_copies9 a h).2.2.2.1
  have c292 := (privateBatchWrapper8_copies9 a h).2.2.2.2.1
  have c293 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.1
  have c294 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.1
  have c295 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c288, k3, ← c289, k3, ← c290] at e_7_6
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c297, ← c298, k3, ← c299] at e_7_7
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c291, ← c292, ← c293] at e_8_0
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c294, ← c295, ← c296] at e_8_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_0
    linear_combination c300.trans k2 - hc
  · have hc := e_7_7
    simp only [e_7_6, e_8_1] at hc
    linear_combination c301.trans k2 - hc

theorem privateBatchWrapper8_f31 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 54)) (a (.virt 11)) (a (.virt 322)) (a (.virt 323)) := by
  have c302 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c302, k3, ← c303, k3, ← c304] at e_7_8
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c311, ← c312, k3, ← c313] at e_7_9
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c305, ← c306, ← c307] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c308, ← c309, ← c310] at e_8_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_2
    linear_combination c314.trans k2 - hc
  · have hc := e_7_9
    simp only [e_7_8, e_8_3] at hc
    linear_combination c315.trans k2 - hc

theorem privateBatchWrapper8_f32 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 8 19) = band (a (.virt 316)) (a (.virt 318)) := by
  have c316 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c316, ← c317, ← c318] at e_8_4
  have hr := e_8_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f33 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 8 23) = band (a (.virt 320)) (a (.virt 322)) := by
  have c319 := (privateBatchWrapper8_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (privateBatchWrapper8_copies10 a h).1
  have c321 := (privateBatchWrapper8_copies10 a h).2.1
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c319, ← c320, ← c321] at e_8_5
  have hr := e_8_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f34 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 8 27) = band (a (.wire 8 19)) (a (.wire 8 23)) := by
  have c322 := (privateBatchWrapper8_copies10 a h).2.2.1
  have c323 := (privateBatchWrapper8_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper8_copies10 a h).2.2.2.2.1
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c322, ← c323, ← c324] at e_8_6
  have hr := e_8_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f35 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 62)) (a (.virt 11)) (a (.virt 324)) (a (.virt 325)) := by
  have c325 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.1
  have c326 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.1
  have c327 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c325, k3, ← c326, k3, ← c327] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c334, ← c335, k3, ← c336] at e_7_11
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c328, ← c329, ← c330] at e_8_7
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c331, ← c332, ← c333] at e_8_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_7
    linear_combination c337.trans k2 - hc
  · have hc := e_7_11
    simp only [e_7_10, e_8_8] at hc
    linear_combination c338.trans k2 - hc

theorem privateBatchWrapper8_f36 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 63)) (a (.virt 11)) (a (.virt 326)) (a (.virt 327)) := by
  have c339 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (privateBatchWrapper8_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (privateBatchWrapper8_copies11 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c339, k3, ← c340, k3, ← c341] at e_7_12
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c348, ← c349, k3, ← c350] at e_7_13
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c342, ← c343, ← c344] at e_8_9
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c345, ← c346, ← c347] at e_8_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_9
    linear_combination c351.trans k2 - hc
  · have hc := e_7_13
    simp only [e_7_12, e_8_10] at hc
    linear_combination c352.trans k2 - hc

theorem privateBatchWrapper8_f37 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 64)) (a (.virt 11)) (a (.virt 328)) (a (.virt 329)) := by
  have c353 := (privateBatchWrapper8_copies11 a h).2.1
  have c354 := (privateBatchWrapper8_copies11 a h).2.2.1
  have c355 := (privateBatchWrapper8_copies11 a h).2.2.2.1
  have c356 := (privateBatchWrapper8_copies11 a h).2.2.2.2.1
  have c357 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.1
  have c358 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.1
  have c359 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have c362 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c353, k3, ← c354, k3, ← c355] at e_7_14
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c356, ← c357, ← c358] at e_8_11
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c359, ← c360, ← c361] at e_8_12
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c362, ← c363, k3, ← c364] at e_9_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_11
    linear_combination c365.trans k2 - hc
  · have hc := e_9_0
    simp only [e_7_14, e_8_12] at hc
    linear_combination c366.trans k2 - hc

theorem privateBatchWrapper8_f38 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 65)) (a (.virt 11)) (a (.virt 330)) (a (.virt 331)) := by
  have c367 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c370, ← c371, ← c372] at e_8_13
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c373, ← c374, ← c375] at e_8_14
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c367, k3, ← c368, k3, ← c369] at e_9_1
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c376, ← c377, k3, ← c378] at e_9_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_13
    linear_combination c379.trans k2 - hc
  · have hc := e_9_2
    simp only [e_9_1, e_8_14] at hc
    linear_combination c380.trans k2 - hc

theorem privateBatchWrapper8_f39 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 10 3) = band (a (.virt 324)) (a (.virt 326)) := by
  have c381 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (privateBatchWrapper8_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c381, ← c382, ← c383] at e_10_0
  have hr := e_10_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f40 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 10 7) = band (a (.virt 328)) (a (.virt 330)) := by
  have c384 := (privateBatchWrapper8_copies12 a h).1
  have c385 := (privateBatchWrapper8_copies12 a h).2.1
  have c386 := (privateBatchWrapper8_copies12 a h).2.2.1
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c384, ← c385, ← c386] at e_10_1
  have hr := e_10_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f41 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 10 11) = band (a (.wire 10 3)) (a (.wire 10 7)) := by
  have c387 := (privateBatchWrapper8_copies12 a h).2.2.2.1
  have c388 := (privateBatchWrapper8_copies12 a h).2.2.2.2.1
  have c389 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.1
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c387, ← c388, ← c389] at e_10_2
  have hr := e_10_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f42 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 73)) (a (.virt 11)) (a (.virt 332)) (a (.virt 333)) := by
  have c390 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.1
  have c391 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c390, k3, ← c391, k3, ← c392] at e_9_3
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c399, ← c400, k3, ← c401] at e_9_4
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c393, ← c394, ← c395] at e_10_3
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c396, ← c397, ← c398] at e_10_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_3
    linear_combination c402.trans k2 - hc
  · have hc := e_9_4
    simp only [e_9_3, e_10_4] at hc
    linear_combination c403.trans k2 - hc

theorem privateBatchWrapper8_f43 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 74)) (a (.virt 11)) (a (.virt 334)) (a (.virt 335)) := by
  have c404 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c405 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (privateBatchWrapper8_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (privateBatchWrapper8_copies13 a h).1
  have c417 := (privateBatchWrapper8_copies13 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c404, k3, ← c405, k3, ← c406] at e_9_5
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c413, ← c414, k3, ← c415] at e_9_6
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c407, ← c408, ← c409] at e_10_5
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c410, ← c411, ← c412] at e_10_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_5
    linear_combination c416.trans k2 - hc
  · have hc := e_9_6
    simp only [e_9_5, e_10_6] at hc
    linear_combination c417.trans k2 - hc

theorem privateBatchWrapper8_f44 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 75)) (a (.virt 11)) (a (.virt 336)) (a (.virt 337)) := by
  have c418 := (privateBatchWrapper8_copies13 a h).2.2.1
  have c419 := (privateBatchWrapper8_copies13 a h).2.2.2.1
  have c420 := (privateBatchWrapper8_copies13 a h).2.2.2.2.1
  have c421 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.1
  have c422 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.1
  have c423 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c418, k3, ← c419, k3, ← c420] at e_9_7
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c427, ← c428, k3, ← c429] at e_9_8
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c421, ← c422, ← c423] at e_10_7
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c424, ← c425, ← c426] at e_10_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_7
    linear_combination c430.trans k2 - hc
  · have hc := e_9_8
    simp only [e_9_7, e_10_8] at hc
    linear_combination c431.trans k2 - hc

theorem privateBatchWrapper8_f45 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 76)) (a (.virt 11)) (a (.virt 338)) (a (.virt 339)) := by
  have c432 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c432, k3, ← c433, k3, ← c434] at e_9_9
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c441, ← c442, k3, ← c443] at e_9_10
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c435, ← c436, ← c437] at e_10_9
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c438, ← c439, ← c440] at e_10_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_9
    linear_combination c444.trans k2 - hc
  · have hc := e_9_10
    simp only [e_9_9, e_10_10] at hc
    linear_combination c445.trans k2 - hc

theorem privateBatchWrapper8_f46 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 10 47) = band (a (.virt 332)) (a (.virt 334)) := by
  have c446 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (privateBatchWrapper8_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (privateBatchWrapper8_copies14 a h).1
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c446, ← c447, ← c448] at e_10_11
  have hr := e_10_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f47 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 10 51) = band (a (.virt 336)) (a (.virt 338)) := by
  have c449 := (privateBatchWrapper8_copies14 a h).2.1
  have c450 := (privateBatchWrapper8_copies14 a h).2.2.1
  have c451 := (privateBatchWrapper8_copies14 a h).2.2.2.1
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c449, ← c450, ← c451] at e_10_12
  have hr := e_10_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f48 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 10 55) = band (a (.wire 10 47)) (a (.wire 10 51)) := by
  have c452 := (privateBatchWrapper8_copies14 a h).2.2.2.2.1
  have c453 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.1
  have c454 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.1
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c452, ← c453, ← c454] at e_10_13
  have hr := e_10_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f49 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 84)) (a (.virt 11)) (a (.virt 340)) (a (.virt 341)) := by
  have c455 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c455, k3, ← c456, k3, ← c457] at e_9_11
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c464, ← c465, k3, ← c466] at e_9_12
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c458, ← c459, ← c460] at e_10_14
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c461, ← c462, ← c463] at e_11_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_14
    linear_combination c467.trans k2 - hc
  · have hc := e_9_12
    simp only [e_9_11, e_11_0] at hc
    linear_combination c468.trans k2 - hc

theorem privateBatchWrapper8_f50 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 85)) (a (.virt 11)) (a (.virt 342)) (a (.virt 343)) := by
  have c469 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (privateBatchWrapper8_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (privateBatchWrapper8_copies15 a h).1
  have c481 := (privateBatchWrapper8_copies15 a h).2.1
  have c482 := (privateBatchWrapper8_copies15 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c469, k3, ← c470, k3, ← c471] at e_9_13
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c478, ← c479, k3, ← c480] at e_9_14
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c472, ← c473, ← c474] at e_11_1
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c475, ← c476, ← c477] at e_11_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_1
    linear_combination c481.trans k2 - hc
  · have hc := e_9_14
    simp only [e_9_13, e_11_2] at hc
    linear_combination c482.trans k2 - hc

theorem privateBatchWrapper8_f51 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 86)) (a (.virt 11)) (a (.virt 344)) (a (.virt 345)) := by
  have c483 := (privateBatchWrapper8_copies15 a h).2.2.2.1
  have c484 := (privateBatchWrapper8_copies15 a h).2.2.2.2.1
  have c485 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.1
  have c486 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.1
  have c487 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c486, ← c487, ← c488] at e_11_3
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c489, ← c490, ← c491] at e_11_4
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c483, k3, ← c484, k3, ← c485] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c492, ← c493, k3, ← c494] at e_12_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_3
    linear_combination c495.trans k2 - hc
  · have hc := e_12_1
    simp only [e_12_0, e_11_4] at hc
    linear_combination c496.trans k2 - hc

theorem privateBatchWrapper8_f52 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 87)) (a (.virt 11)) (a (.virt 346)) (a (.virt 347)) := by
  have c497 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c500, ← c501, ← c502] at e_11_5
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c503, ← c504, ← c505] at e_11_6
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c497, k3, ← c498, k3, ← c499] at e_12_2
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c506, ← c507, k3, ← c508] at e_12_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_5
    linear_combination c509.trans k2 - hc
  · have hc := e_12_3
    simp only [e_12_2, e_11_6] at hc
    linear_combination c510.trans k2 - hc

theorem privateBatchWrapper8_f53 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 31) = band (a (.virt 340)) (a (.virt 342)) := by
  have c511 := (privateBatchWrapper8_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (privateBatchWrapper8_copies16 a h).1
  have c513 := (privateBatchWrapper8_copies16 a h).2.1
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c511, ← c512, ← c513] at e_11_7
  have hr := e_11_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f54 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 35) = band (a (.virt 344)) (a (.virt 346)) := by
  have c514 := (privateBatchWrapper8_copies16 a h).2.2.1
  have c515 := (privateBatchWrapper8_copies16 a h).2.2.2.1
  have c516 := (privateBatchWrapper8_copies16 a h).2.2.2.2.1
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c514, ← c515, ← c516] at e_11_8
  have hr := e_11_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f55 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 39) = band (a (.wire 11 31)) (a (.wire 11 35)) := by
  have c517 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.1
  have c518 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.1
  have c519 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.1
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c517, ← c518, ← c519] at e_11_9
  have hr := e_11_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f56 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 19) = bnot (a (.wire 3 31)) := by
  have c520 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c520, k3, ← c521, k3, ← c522] at e_12_4
  have hr := e_12_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f57 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 12) = bnot (a (.virt 11)) := by
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [bnot, k2, k3]
  ring

theorem privateBatchWrapper8_f58 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 19) = band (a (.wire 12 19)) (a (.virt 12)) := by
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [band, k3]
  ring

theorem privateBatchWrapper8_f59 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 23) = bselect (a (.wire 12 19)) (a (.virt 5)) (a (.virt 11)) := by
  have c523 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c523, ← c524, ← c525, k2] at e_12_5
  have hr := e_12_5
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f60 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 27) = bselect (a (.wire 12 19)) (a (.virt 6)) (a (.virt 11)) := by
  have c526 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c526, ← c527, ← c528, k2] at e_12_6
  have hr := e_12_6
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f61 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 31) = bselect (a (.wire 12 19)) (a (.virt 7)) (a (.virt 11)) := by
  have c529 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c529, ← c530, ← c531, k2] at e_12_7
  have hr := e_12_7
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f62 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 35) = bselect (a (.wire 12 19)) (a (.virt 8)) (a (.virt 11)) := by
  have c532 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c532, ← c533, ← c534, k2] at e_12_8
  have hr := e_12_8
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f63 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 39) = bselect (a (.wire 12 19)) (a (.virt 9)) (a (.virt 11)) := by
  have c535 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c535, ← c536, ← c537, k2] at e_12_9
  have hr := e_12_9
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f64 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 19) = bor (a (.virt 11)) (a (.wire 12 19)) := by
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [bor, k2]
  ring

theorem privateBatchWrapper8_f65 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 43) = bnot (a (.wire 4 15)) := by
  have c538 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c538, k3, ← c539, k3, ← c540] at e_12_10
  have hr := e_12_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f66 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 47) = bnot (a (.wire 12 19)) := by
  have c541 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c541, k3, ← c542, k3, ← c543] at e_12_11
  have hr := e_12_11
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f67 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 43) = band (a (.wire 12 43)) (a (.wire 12 47)) := by
  have c544 := (privateBatchWrapper8_copies17 a h).1
  have c545 := (privateBatchWrapper8_copies17 a h).2.1
  have c546 := (privateBatchWrapper8_copies17 a h).2.2.1
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c544, ← c545, ← c546] at e_11_10
  have hr := e_11_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f68 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 55) = bselect (a (.wire 11 43)) (a (.virt 18)) (a (.wire 12 23)) := by
  have c547 := (privateBatchWrapper8_copies17 a h).2.2.2.1
  have c548 := (privateBatchWrapper8_copies17 a h).2.2.2.2.1
  have c549 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.1
  have c550 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.1
  have c551 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.1
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c547, ← c548, ← c549] at e_12_12
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c550, ← c551, ← c552] at e_12_13
  have hr := e_12_13
  simp only [e_12_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f69 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 3) = bselect (a (.wire 11 43)) (a (.virt 19)) (a (.wire 12 27)) := by
  have c553 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c553, ← c554, ← c555] at e_12_14
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c556, ← c557, ← c558] at e_13_0
  have hr := e_13_0
  simp only [e_12_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f70 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 11) = bselect (a (.wire 11 43)) (a (.virt 20)) (a (.wire 12 31)) := by
  have c559 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c559, ← c560, ← c561] at e_13_1
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c562, ← c563, ← c564] at e_13_2
  have hr := e_13_2
  simp only [e_13_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f71 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 19) = bselect (a (.wire 11 43)) (a (.virt 21)) (a (.wire 12 35)) := by
  have c565 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c565, ← c566, ← c567] at e_13_3
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c568, ← c569, ← c570] at e_13_4
  have hr := e_13_4
  simp only [e_13_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f72 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 27) = bselect (a (.wire 11 43)) (a (.virt 22)) (a (.wire 12 39)) := by
  have c571 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (privateBatchWrapper8_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (privateBatchWrapper8_copies18 a h).1
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c571, ← c572, ← c573] at e_13_5
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c574, ← c575, ← c576] at e_13_6
  have hr := e_13_6
  simp only [e_13_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f73 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 0 59) = bor (a (.wire 12 19)) (a (.wire 12 43)) := by
  have c577 := (privateBatchWrapper8_copies18 a h).2.1
  have c578 := (privateBatchWrapper8_copies18 a h).2.2.1
  have c579 := (privateBatchWrapper8_copies18 a h).2.2.2.1
  have c580 := (privateBatchWrapper8_copies18 a h).2.2.2.2.1
  have c581 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.1
  have c582 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_14 := arithEq_of_rows h (row := 0) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c580, ← c581, k3, ← c582] at e_0_14
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c577, ← c578, ← c579] at e_14_0
  have hr := e_0_14
  simp only [e_14_0] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f74 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 31) = bnot (a (.wire 4 59)) := by
  have c583 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c583, k3, ← c584, k3, ← c585] at e_13_7
  have hr := e_13_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f75 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 35) = bnot (a (.wire 0 59)) := by
  have c586 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c586, k3, ← c587, k3, ← c588] at e_13_8
  have hr := e_13_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f76 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 47) = band (a (.wire 13 31)) (a (.wire 13 35)) := by
  have c589 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c589, ← c590, ← c591] at e_11_11
  have hr := e_11_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f77 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 43) = bselect (a (.wire 11 47)) (a (.virt 29)) (a (.wire 12 55)) := by
  have c592 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c592, ← c593, ← c594] at e_13_9
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c595, ← c596, ← c597] at e_13_10
  have hr := e_13_10
  simp only [e_13_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f78 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 51) = bselect (a (.wire 11 47)) (a (.virt 30)) (a (.wire 13 3)) := by
  have c598 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c600 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c598, ← c599, ← c600] at e_13_11
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c601, ← c602, ← c603] at e_13_12
  have hr := e_13_12
  simp only [e_13_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f79 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 59) = bselect (a (.wire 11 47)) (a (.virt 31)) (a (.wire 13 11)) := by
  have c604 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (privateBatchWrapper8_copies19 a h).1
  have c609 := (privateBatchWrapper8_copies19 a h).2.1
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c604, ← c605, ← c606] at e_13_13
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c607, ← c608, ← c609] at e_13_14
  have hr := e_13_14
  simp only [e_13_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f80 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 7) = bselect (a (.wire 11 47)) (a (.virt 32)) (a (.wire 13 19)) := by
  have c610 := (privateBatchWrapper8_copies19 a h).2.2.1
  have c611 := (privateBatchWrapper8_copies19 a h).2.2.2.1
  have c612 := (privateBatchWrapper8_copies19 a h).2.2.2.2.1
  have c613 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.1
  have c614 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.1
  have c615 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.1
  have e_15_0 := arithEq_of_rows h (row := 15) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c610, ← c611, ← c612] at e_15_0
  have e_15_1 := arithEq_of_rows h (row := 15) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c613, ← c614, ← c615] at e_15_1
  have hr := e_15_1
  simp only [e_15_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f81 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 15) = bselect (a (.wire 11 47)) (a (.virt 33)) (a (.wire 13 27)) := by
  have c616 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_2 := arithEq_of_rows h (row := 15) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c616, ← c617, ← c618] at e_15_2
  have e_15_3 := arithEq_of_rows h (row := 15) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c619, ← c620, ← c621] at e_15_3
  have hr := e_15_3
  simp only [e_15_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f82 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 3) = bor (a (.wire 0 59)) (a (.wire 13 31)) := by
  have c622 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c622, ← c623, ← c624] at e_14_1
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c625, ← c626, k3, ← c627] at e_16_0
  have hr := e_16_0
  simp only [e_14_1] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f83 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 19) = bnot (a (.wire 6 43)) := by
  have c628 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c628, k3, ← c629, k3, ← c630] at e_15_4
  have hr := e_15_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f84 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 23) = bnot (a (.wire 16 3)) := by
  have c631 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_5 := arithEq_of_rows h (row := 15) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c631, k3, ← c632, k3, ← c633] at e_15_5
  have hr := e_15_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f85 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 51) = band (a (.wire 15 19)) (a (.wire 15 23)) := by
  have c634 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c634, ← c635, ← c636] at e_11_12
  have hr := e_11_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f86 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 31) = bselect (a (.wire 11 51)) (a (.virt 40)) (a (.wire 13 43)) := by
  have c637 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (privateBatchWrapper8_copies20 a h).1
  have c641 := (privateBatchWrapper8_copies20 a h).2.1
  have c642 := (privateBatchWrapper8_copies20 a h).2.2.1
  have e_15_6 := arithEq_of_rows h (row := 15) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c637, ← c638, ← c639] at e_15_6
  have e_15_7 := arithEq_of_rows h (row := 15) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_7
  simp only [← c640, ← c641, ← c642] at e_15_7
  have hr := e_15_7
  simp only [e_15_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f87 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 39) = bselect (a (.wire 11 51)) (a (.virt 41)) (a (.wire 13 51)) := by
  have c643 := (privateBatchWrapper8_copies20 a h).2.2.2.1
  have c644 := (privateBatchWrapper8_copies20 a h).2.2.2.2.1
  have c645 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.1
  have c646 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.1
  have c647 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.1
  have e_15_8 := arithEq_of_rows h (row := 15) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_8
  simp only [← c643, ← c644, ← c645] at e_15_8
  have e_15_9 := arithEq_of_rows h (row := 15) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_9
  simp only [← c646, ← c647, ← c648] at e_15_9
  have hr := e_15_9
  simp only [e_15_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f88 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 47) = bselect (a (.wire 11 51)) (a (.virt 42)) (a (.wire 13 59)) := by
  have c649 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_10 := arithEq_of_rows h (row := 15) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_10
  simp only [← c649, ← c650, ← c651] at e_15_10
  have e_15_11 := arithEq_of_rows h (row := 15) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_11
  simp only [← c652, ← c653, ← c654] at e_15_11
  have hr := e_15_11
  simp only [e_15_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f89 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 55) = bselect (a (.wire 11 51)) (a (.virt 43)) (a (.wire 15 7)) := by
  have c655 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_12 := arithEq_of_rows h (row := 15) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_12
  simp only [← c655, ← c656, ← c657] at e_15_12
  have e_15_13 := arithEq_of_rows h (row := 15) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_13
  simp only [← c658, ← c659, ← c660] at e_15_13
  have hr := e_15_13
  simp only [e_15_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f90 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 3) = bselect (a (.wire 11 51)) (a (.virt 44)) (a (.wire 15 15)) := by
  have c661 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_14 := arithEq_of_rows h (row := 15) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_14
  simp only [← c661, ← c662, ← c663] at e_15_14
  have e_17_0 := arithEq_of_rows h (row := 17) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c664, ← c665, ← c666] at e_17_0
  have hr := e_17_0
  simp only [e_15_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f91 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 7) = bor (a (.wire 16 3)) (a (.wire 15 19)) := by
  have c667 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (privateBatchWrapper8_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (privateBatchWrapper8_copies21 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c667, ← c668, ← c669] at e_14_2
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c670, ← c671, k3, ← c672] at e_16_1
  have hr := e_16_1
  simp only [e_14_2] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f92 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 7) = bnot (a (.wire 8 27)) := by
  have c673 := (privateBatchWrapper8_copies21 a h).2.1
  have c674 := (privateBatchWrapper8_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper8_copies21 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c673, k3, ← c674, k3, ← c675] at e_17_1
  have hr := e_17_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f93 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 11) = bnot (a (.wire 16 7)) := by
  have c676 := (privateBatchWrapper8_copies21 a h).2.2.2.2.1
  have c677 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.1
  have c678 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_2 := arithEq_of_rows h (row := 17) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c676, k3, ← c677, k3, ← c678] at e_17_2
  have hr := e_17_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f94 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 55) = band (a (.wire 17 7)) (a (.wire 17 11)) := by
  have c679 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c679, ← c680, ← c681] at e_11_13
  have hr := e_11_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f95 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 19) = bselect (a (.wire 11 55)) (a (.virt 51)) (a (.wire 15 31)) := by
  have c682 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_3 := arithEq_of_rows h (row := 17) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c682, ← c683, ← c684] at e_17_3
  have e_17_4 := arithEq_of_rows h (row := 17) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c685, ← c686, ← c687] at e_17_4
  have hr := e_17_4
  simp only [e_17_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f96 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 27) = bselect (a (.wire 11 55)) (a (.virt 52)) (a (.wire 15 39)) := by
  have c688 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_5 := arithEq_of_rows h (row := 17) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c688, ← c689, ← c690] at e_17_5
  have e_17_6 := arithEq_of_rows h (row := 17) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c691, ← c692, ← c693] at e_17_6
  have hr := e_17_6
  simp only [e_17_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f97 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 35) = bselect (a (.wire 11 55)) (a (.virt 53)) (a (.wire 15 47)) := by
  have c694 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_7 := arithEq_of_rows h (row := 17) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c694, ← c695, ← c696] at e_17_7
  have e_17_8 := arithEq_of_rows h (row := 17) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c697, ← c698, ← c699] at e_17_8
  have hr := e_17_8
  simp only [e_17_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f98 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 43) = bselect (a (.wire 11 55)) (a (.virt 54)) (a (.wire 15 55)) := by
  have c700 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (privateBatchWrapper8_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (privateBatchWrapper8_copies22 a h).1
  have c705 := (privateBatchWrapper8_copies22 a h).2.1
  have e_17_9 := arithEq_of_rows h (row := 17) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c700, ← c701, ← c702] at e_17_9
  have e_17_10 := arithEq_of_rows h (row := 17) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c703, ← c704, ← c705] at e_17_10
  have hr := e_17_10
  simp only [e_17_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f99 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 51) = bselect (a (.wire 11 55)) (a (.virt 55)) (a (.wire 17 3)) := by
  have c706 := (privateBatchWrapper8_copies22 a h).2.2.1
  have c707 := (privateBatchWrapper8_copies22 a h).2.2.2.1
  have c708 := (privateBatchWrapper8_copies22 a h).2.2.2.2.1
  have c709 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.1
  have c710 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.1
  have c711 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.1
  have e_17_11 := arithEq_of_rows h (row := 17) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c706, ← c707, ← c708] at e_17_11
  have e_17_12 := arithEq_of_rows h (row := 17) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c709, ← c710, ← c711] at e_17_12
  have hr := e_17_12
  simp only [e_17_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f100 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 11) = bor (a (.wire 16 7)) (a (.wire 17 7)) := by
  have c712 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c712, ← c713, ← c714] at e_14_3
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c715, ← c716, k3, ← c717] at e_16_2
  have hr := e_16_2
  simp only [e_14_3] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f101 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f102 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 59) = bnot (a (.wire 16 11)) := by
  have c721 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_14 := arithEq_of_rows h (row := 17) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c721, k3, ← c722, k3, ← c723] at e_17_14
  have hr := e_17_14
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f103 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 11 59) = band (a (.wire 17 55)) (a (.wire 17 59)) := by
  have c724 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c724, ← c725, ← c726] at e_11_14
  have hr := e_11_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f104 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 7) = bselect (a (.wire 11 59)) (a (.virt 62)) (a (.wire 17 19)) := by
  have c727 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_0 := arithEq_of_rows h (row := 18) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c727, ← c728, ← c729] at e_18_0
  have e_18_1 := arithEq_of_rows h (row := 18) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c730, ← c731, ← c732] at e_18_1
  have hr := e_18_1
  simp only [e_18_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f105 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 15) = bselect (a (.wire 11 59)) (a (.virt 63)) (a (.wire 17 27)) := by
  have c733 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c736 := (privateBatchWrapper8_copies23 a h).1
  have c737 := (privateBatchWrapper8_copies23 a h).2.1
  have c738 := (privateBatchWrapper8_copies23 a h).2.2.1
  have e_18_2 := arithEq_of_rows h (row := 18) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c733, ← c734, ← c735] at e_18_2
  have e_18_3 := arithEq_of_rows h (row := 18) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c736, ← c737, ← c738] at e_18_3
  have hr := e_18_3
  simp only [e_18_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f106 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 23) = bselect (a (.wire 11 59)) (a (.virt 64)) (a (.wire 17 35)) := by
  have c739 := (privateBatchWrapper8_copies23 a h).2.2.2.1
  have c740 := (privateBatchWrapper8_copies23 a h).2.2.2.2.1
  have c741 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.1
  have c742 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.1
  have c743 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.1
  have e_18_4 := arithEq_of_rows h (row := 18) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c739, ← c740, ← c741] at e_18_4
  have e_18_5 := arithEq_of_rows h (row := 18) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c742, ← c743, ← c744] at e_18_5
  have hr := e_18_5
  simp only [e_18_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f107 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 31) = bselect (a (.wire 11 59)) (a (.virt 65)) (a (.wire 17 43)) := by
  have c745 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_6 := arithEq_of_rows h (row := 18) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c745, ← c746, ← c747] at e_18_6
  have e_18_7 := arithEq_of_rows h (row := 18) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c748, ← c749, ← c750] at e_18_7
  have hr := e_18_7
  simp only [e_18_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f108 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 39) = bselect (a (.wire 11 59)) (a (.virt 66)) (a (.wire 17 51)) := by
  have c751 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c753 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_8 := arithEq_of_rows h (row := 18) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c751, ← c752, ← c753] at e_18_8
  have e_18_9 := arithEq_of_rows h (row := 18) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c754, ← c755, ← c756] at e_18_9
  have hr := e_18_9
  simp only [e_18_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f109 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 15) = bor (a (.wire 16 11)) (a (.wire 17 55)) := by
  have c757 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c759 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c757, ← c758, ← c759] at e_14_4
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c760, ← c761, k3, ← c762] at e_16_3
  have hr := e_16_3
  simp only [e_14_4] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f110 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f111 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 47) = bnot (a (.wire 16 15)) := by
  have c766 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (privateBatchWrapper8_copies24 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_11 := arithEq_of_rows h (row := 18) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c766, k3, ← c767, k3, ← c768] at e_18_11
  have hr := e_18_11
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f112 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 19 3) = band (a (.wire 18 43)) (a (.wire 18 47)) := by
  have c769 := (privateBatchWrapper8_copies24 a h).2.1
  have c770 := (privateBatchWrapper8_copies24 a h).2.2.1
  have c771 := (privateBatchWrapper8_copies24 a h).2.2.2.1
  have e_19_0 := arithEq_of_rows h (row := 19) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c769, ← c770, ← c771] at e_19_0
  have hr := e_19_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f113 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 55) = bselect (a (.wire 19 3)) (a (.virt 73)) (a (.wire 18 7)) := by
  have c772 := (privateBatchWrapper8_copies24 a h).2.2.2.2.1
  have c773 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.1
  have c774 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.1
  have c775 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have e_18_12 := arithEq_of_rows h (row := 18) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c772, ← c773, ← c774] at e_18_12
  have e_18_13 := arithEq_of_rows h (row := 18) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c775, ← c776, ← c777] at e_18_13
  have hr := e_18_13
  simp only [e_18_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f114 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 3) = bselect (a (.wire 19 3)) (a (.virt 74)) (a (.wire 18 15)) := by
  have c778 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c781 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_14 := arithEq_of_rows h (row := 18) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c778, ← c779, ← c780] at e_18_14
  have e_20_0 := arithEq_of_rows h (row := 20) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_0
  simp only [← c781, ← c782, ← c783] at e_20_0
  have hr := e_20_0
  simp only [e_18_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f115 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 11) = bselect (a (.wire 19 3)) (a (.virt 75)) (a (.wire 18 23)) := by
  have c784 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_20_1 := arithEq_of_rows h (row := 20) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_1
  simp only [← c784, ← c785, ← c786] at e_20_1
  have e_20_2 := arithEq_of_rows h (row := 20) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_2
  simp only [← c787, ← c788, ← c789] at e_20_2
  have hr := e_20_2
  simp only [e_20_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f116 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 19) = bselect (a (.wire 19 3)) (a (.virt 76)) (a (.wire 18 31)) := by
  have c790 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_20_3 := arithEq_of_rows h (row := 20) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_3
  simp only [← c790, ← c791, ← c792] at e_20_3
  have e_20_4 := arithEq_of_rows h (row := 20) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_4
  simp only [← c793, ← c794, ← c795] at e_20_4
  have hr := e_20_4
  simp only [e_20_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f117 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 27) = bselect (a (.wire 19 3)) (a (.virt 77)) (a (.wire 18 39)) := by
  have c796 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (privateBatchWrapper8_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (privateBatchWrapper8_copies25 a h).1
  have c801 := (privateBatchWrapper8_copies25 a h).2.1
  have e_20_5 := arithEq_of_rows h (row := 20) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_5
  simp only [← c796, ← c797, ← c798] at e_20_5
  have e_20_6 := arithEq_of_rows h (row := 20) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_6
  simp only [← c799, ← c800, ← c801] at e_20_6
  have hr := e_20_6
  simp only [e_20_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f118 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 19) = bor (a (.wire 16 15)) (a (.wire 18 43)) := by
  have c802 := (privateBatchWrapper8_copies25 a h).2.2.1
  have c803 := (privateBatchWrapper8_copies25 a h).2.2.2.1
  have c804 := (privateBatchWrapper8_copies25 a h).2.2.2.2.1
  have c805 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.1
  have c806 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.1
  have c807 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c802, ← c803, ← c804] at e_14_5
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c805, ← c806, k3, ← c807] at e_16_4
  have hr := e_16_4
  simp only [e_14_5] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f119 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f120 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 35) = bnot (a (.wire 16 19)) := by
  have c811 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_8 := arithEq_of_rows h (row := 20) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_8
  simp only [← c811, k3, ← c812, k3, ← c813] at e_20_8
  have hr := e_20_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f121 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 19 7) = band (a (.wire 20 31)) (a (.wire 20 35)) := by
  have c814 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_1 := arithEq_of_rows h (row := 19) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c814, ← c815, ← c816] at e_19_1
  have hr := e_19_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f122 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 43) = bselect (a (.wire 19 7)) (a (.virt 84)) (a (.wire 18 55)) := by
  have c817 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_20_9 := arithEq_of_rows h (row := 20) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_9
  simp only [← c817, ← c818, ← c819] at e_20_9
  have e_20_10 := arithEq_of_rows h (row := 20) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_10
  simp only [← c820, ← c821, ← c822] at e_20_10
  have hr := e_20_10
  simp only [e_20_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f123 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 51) = bselect (a (.wire 19 7)) (a (.virt 85)) (a (.wire 20 3)) := by
  have c823 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_20_11 := arithEq_of_rows h (row := 20) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_11
  simp only [← c823, ← c824, ← c825] at e_20_11
  have e_20_12 := arithEq_of_rows h (row := 20) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_12
  simp only [← c826, ← c827, ← c828] at e_20_12
  have hr := e_20_12
  simp only [e_20_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f124 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 59) = bselect (a (.wire 19 7)) (a (.virt 86)) (a (.wire 20 11)) := by
  have c829 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c832 := (privateBatchWrapper8_copies26 a h).1
  have c833 := (privateBatchWrapper8_copies26 a h).2.1
  have c834 := (privateBatchWrapper8_copies26 a h).2.2.1
  have e_20_13 := arithEq_of_rows h (row := 20) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_13
  simp only [← c829, ← c830, ← c831] at e_20_13
  have e_20_14 := arithEq_of_rows h (row := 20) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_14
  simp only [← c832, ← c833, ← c834] at e_20_14
  have hr := e_20_14
  simp only [e_20_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f125 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 21 7) = bselect (a (.wire 19 7)) (a (.virt 87)) (a (.wire 20 19)) := by
  have c835 := (privateBatchWrapper8_copies26 a h).2.2.2.1
  have c836 := (privateBatchWrapper8_copies26 a h).2.2.2.2.1
  have c837 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.1
  have c838 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.1
  have c839 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.1
  have e_21_0 := arithEq_of_rows h (row := 21) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c835, ← c836, ← c837] at e_21_0
  have e_21_1 := arithEq_of_rows h (row := 21) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c838, ← c839, ← c840] at e_21_1
  have hr := e_21_1
  simp only [e_21_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f126 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 21 15) = bselect (a (.wire 19 7)) (a (.virt 88)) (a (.wire 20 27)) := by
  have c841 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have c842 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c844 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c845 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_21_2 := arithEq_of_rows h (row := 21) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c841, ← c842, ← c843] at e_21_2
  have e_21_3 := arithEq_of_rows h (row := 21) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c844, ← c845, ← c846] at e_21_3
  have hr := e_21_3
  simp only [e_21_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f127 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 23) = bor (a (.wire 16 19)) (a (.wire 20 31)) := by
  have c847 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c847, ← c848, ← c849] at e_14_6
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c850, ← c851, k3, ← c852] at e_16_5
  have hr := e_16_5
  simp only [e_14_6] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f128 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 5)) (a (.wire 20 43)) (a (.virt 348)) (a (.virt 349)) := by
  have c853 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c860 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c863 := (privateBatchWrapper8_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (privateBatchWrapper8_copies27 a h).1
  have c865 := (privateBatchWrapper8_copies27 a h).2.1
  have c866 := (privateBatchWrapper8_copies27 a h).2.2.1
  have c867 := (privateBatchWrapper8_copies27 a h).2.2.2.1
  have c868 := (privateBatchWrapper8_copies27 a h).2.2.2.2.1
  have c869 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_19_2 := arithEq_of_rows h (row := 19) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c859, ← c860, ← c861] at e_19_2
  have e_19_3 := arithEq_of_rows h (row := 19) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c862, ← c863, ← c864] at e_19_3
  have e_21_4 := arithEq_of_rows h (row := 21) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c853, k3, ← c854, k3, ← c855] at e_21_4
  have e_21_5 := arithEq_of_rows h (row := 21) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c856, ← c857, k3, ← c858] at e_21_5
  have e_21_6 := arithEq_of_rows h (row := 21) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c865, ← c866, k3, ← c867] at e_21_6
  refine ⟨?_, ?_⟩
  · have hc := e_19_2
    simp only [e_21_5] at hc
    linear_combination c868.trans k2 - hc
  · have hc := e_21_6
    simp only [e_21_4, e_19_3, e_21_5] at hc
    linear_combination c869.trans k2 - hc

theorem privateBatchWrapper8_f129 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 6)) (a (.wire 20 51)) (a (.virt 350)) (a (.virt 351)) := by
  have c870 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.1
  have c871 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.1
  have c873 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c876 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c880 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c883 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_19_4 := arithEq_of_rows h (row := 19) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c876, ← c877, ← c878] at e_19_4
  have e_19_5 := arithEq_of_rows h (row := 19) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c879, ← c880, ← c881] at e_19_5
  have e_21_7 := arithEq_of_rows h (row := 21) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c870, k3, ← c871, k3, ← c872] at e_21_7
  have e_21_8 := arithEq_of_rows h (row := 21) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c873, ← c874, k3, ← c875] at e_21_8
  have e_21_9 := arithEq_of_rows h (row := 21) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c882, ← c883, k3, ← c884] at e_21_9
  refine ⟨?_, ?_⟩
  · have hc := e_19_4
    simp only [e_21_8] at hc
    linear_combination c885.trans k2 - hc
  · have hc := e_21_9
    simp only [e_21_7, e_19_5, e_21_8] at hc
    linear_combination c886.trans k2 - hc

theorem privateBatchWrapper8_f130 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 7)) (a (.wire 20 59)) (a (.virt 352)) (a (.virt 353)) := by
  have c887 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (privateBatchWrapper8_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (privateBatchWrapper8_copies28 a h).1
  have c897 := (privateBatchWrapper8_copies28 a h).2.1
  have c898 := (privateBatchWrapper8_copies28 a h).2.2.1
  have c899 := (privateBatchWrapper8_copies28 a h).2.2.2.1
  have c900 := (privateBatchWrapper8_copies28 a h).2.2.2.2.1
  have c901 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.1
  have c902 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.1
  have c903 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_19_6 := arithEq_of_rows h (row := 19) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c893, ← c894, ← c895] at e_19_6
  have e_19_7 := arithEq_of_rows h (row := 19) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c896, ← c897, ← c898] at e_19_7
  have e_21_10 := arithEq_of_rows h (row := 21) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c887, k3, ← c888, k3, ← c889] at e_21_10
  have e_21_11 := arithEq_of_rows h (row := 21) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c890, ← c891, k3, ← c892] at e_21_11
  have e_21_12 := arithEq_of_rows h (row := 21) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c899, ← c900, k3, ← c901] at e_21_12
  refine ⟨?_, ?_⟩
  · have hc := e_19_6
    simp only [e_21_11] at hc
    linear_combination c902.trans k2 - hc
  · have hc := e_21_12
    simp only [e_21_10, e_19_7, e_21_11] at hc
    linear_combination c903.trans k2 - hc

theorem privateBatchWrapper8_f131 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 8)) (a (.wire 21 7)) (a (.virt 354)) (a (.virt 355)) := by
  have c904 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have c907 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_19_8 := arithEq_of_rows h (row := 19) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c910, ← c911, ← c912] at e_19_8
  have e_19_9 := arithEq_of_rows h (row := 19) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c913, ← c914, ← c915] at e_19_9
  have e_21_13 := arithEq_of_rows h (row := 21) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c904, k3, ← c905, k3, ← c906] at e_21_13
  have e_21_14 := arithEq_of_rows h (row := 21) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c907, ← c908, k3, ← c909] at e_21_14
  have e_22_0 := arithEq_of_rows h (row := 22) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_0
  simp only [← c916, ← c917, k3, ← c918] at e_22_0
  refine ⟨?_, ?_⟩
  · have hc := e_19_8
    simp only [e_21_14] at hc
    linear_combination c919.trans k2 - hc
  · have hc := e_22_0
    simp only [e_21_13, e_19_9, e_21_14] at hc
    linear_combination c920.trans k2 - hc

theorem privateBatchWrapper8_f132 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 19 43) = band (a (.virt 348)) (a (.virt 350)) := by
  have c921 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_10 := arithEq_of_rows h (row := 19) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c921, ← c922, ← c923] at e_19_10
  have hr := e_19_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f133 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 19 47) = band (a (.virt 352)) (a (.virt 354)) := by
  have c924 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c925 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_11 := arithEq_of_rows h (row := 19) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c924, ← c925, ← c926] at e_19_11
  have hr := e_19_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f134 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 19 51) = band (a (.wire 19 43)) (a (.wire 19 47)) := by
  have c927 := (privateBatchWrapper8_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c928 := (privateBatchWrapper8_copies29 a h).1
  have c929 := (privateBatchWrapper8_copies29 a h).2.1
  have e_19_12 := arithEq_of_rows h (row := 19) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c927, ← c928, ← c929] at e_19_12
  have hr := e_19_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f135 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 27) = bor (a (.wire 3 31)) (a (.wire 19 51)) := by
  have c930 := (privateBatchWrapper8_copies29 a h).2.2.1
  have c931 := (privateBatchWrapper8_copies29 a h).2.2.2.1
  have c932 := (privateBatchWrapper8_copies29 a h).2.2.2.2.1
  have c933 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.1
  have c934 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.1
  have c935 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c930, ← c931, ← c932] at e_14_7
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c933, ← c934, k3, ← c935] at e_16_6
  have hr := e_16_6
  simp only [e_14_7] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f136 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 27) = a (.virt 12) := by
  have c936 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.1
  exact c936

theorem privateBatchWrapper8_f138 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 18)) (a (.wire 20 43)) (a (.virt 356)) (a (.virt 357)) := by
  have c938 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c947 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c953 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_19_13 := arithEq_of_rows h (row := 19) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c944, ← c945, ← c946] at e_19_13
  have e_19_14 := arithEq_of_rows h (row := 19) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c947, ← c948, ← c949] at e_19_14
  have e_22_1 := arithEq_of_rows h (row := 22) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_1
  simp only [← c938, k3, ← c939, k3, ← c940] at e_22_1
  have e_22_2 := arithEq_of_rows h (row := 22) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_2
  simp only [← c941, ← c942, k3, ← c943] at e_22_2
  have e_22_3 := arithEq_of_rows h (row := 22) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_3
  simp only [← c950, ← c951, k3, ← c952] at e_22_3
  refine ⟨?_, ?_⟩
  · have hc := e_19_13
    simp only [e_22_2] at hc
    linear_combination c953.trans k2 - hc
  · have hc := e_22_3
    simp only [e_22_1, e_19_14, e_22_2] at hc
    linear_combination c954.trans k2 - hc

theorem privateBatchWrapper8_f139 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 19)) (a (.wire 20 51)) (a (.virt 358)) (a (.virt 359)) := by
  have c955 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c956 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c959 := (privateBatchWrapper8_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (privateBatchWrapper8_copies30 a h).1
  have c961 := (privateBatchWrapper8_copies30 a h).2.1
  have c962 := (privateBatchWrapper8_copies30 a h).2.2.1
  have c963 := (privateBatchWrapper8_copies30 a h).2.2.2.1
  have c964 := (privateBatchWrapper8_copies30 a h).2.2.2.2.1
  have c965 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.1
  have c966 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.1
  have c967 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  have c971 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_22_4 := arithEq_of_rows h (row := 22) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_4
  simp only [← c955, k3, ← c956, k3, ← c957] at e_22_4
  have e_22_5 := arithEq_of_rows h (row := 22) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_5
  simp only [← c958, ← c959, k3, ← c960] at e_22_5
  have e_22_6 := arithEq_of_rows h (row := 22) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_6
  simp only [← c967, ← c968, k3, ← c969] at e_22_6
  have e_23_0 := arithEq_of_rows h (row := 23) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_0
  simp only [← c961, ← c962, ← c963] at e_23_0
  have e_23_1 := arithEq_of_rows h (row := 23) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_1
  simp only [← c964, ← c965, ← c966] at e_23_1
  refine ⟨?_, ?_⟩
  · have hc := e_23_0
    simp only [e_22_5] at hc
    linear_combination c970.trans k2 - hc
  · have hc := e_22_6
    simp only [e_22_4, e_23_1, e_22_5] at hc
    linear_combination c971.trans k2 - hc

theorem privateBatchWrapper8_f140 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 20)) (a (.wire 20 59)) (a (.virt 360)) (a (.virt 361)) := by
  have c972 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c977 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c987 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_22_7 := arithEq_of_rows h (row := 22) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_7
  simp only [← c972, k3, ← c973, k3, ← c974] at e_22_7
  have e_22_8 := arithEq_of_rows h (row := 22) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_8
  simp only [← c975, ← c976, k3, ← c977] at e_22_8
  have e_22_9 := arithEq_of_rows h (row := 22) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_9
  simp only [← c984, ← c985, k3, ← c986] at e_22_9
  have e_23_2 := arithEq_of_rows h (row := 23) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_2
  simp only [← c978, ← c979, ← c980] at e_23_2
  have e_23_3 := arithEq_of_rows h (row := 23) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_3
  simp only [← c981, ← c982, ← c983] at e_23_3
  refine ⟨?_, ?_⟩
  · have hc := e_23_2
    simp only [e_22_8] at hc
    linear_combination c987.trans k2 - hc
  · have hc := e_22_9
    simp only [e_22_7, e_23_3, e_22_8] at hc
    linear_combination c988.trans k2 - hc

theorem privateBatchWrapper8_f141 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 21)) (a (.wire 21 7)) (a (.virt 362)) (a (.virt 363)) := by
  have c989 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c990 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (privateBatchWrapper8_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (privateBatchWrapper8_copies31 a h).1
  have c993 := (privateBatchWrapper8_copies31 a h).2.1
  have c994 := (privateBatchWrapper8_copies31 a h).2.2.1
  have c995 := (privateBatchWrapper8_copies31 a h).2.2.2.1
  have c996 := (privateBatchWrapper8_copies31 a h).2.2.2.2.1
  have c997 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.1
  have c998 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.1
  have c999 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have c1002 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1005 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_22_10 := arithEq_of_rows h (row := 22) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_10
  simp only [← c989, k3, ← c990, k3, ← c991] at e_22_10
  have e_22_11 := arithEq_of_rows h (row := 22) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c992, ← c993, k3, ← c994] at e_22_11
  have e_22_12 := arithEq_of_rows h (row := 22) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_12
  simp only [← c1001, ← c1002, k3, ← c1003] at e_22_12
  have e_23_4 := arithEq_of_rows h (row := 23) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_4
  simp only [← c995, ← c996, ← c997] at e_23_4
  have e_23_5 := arithEq_of_rows h (row := 23) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_5
  simp only [← c998, ← c999, ← c1000] at e_23_5
  refine ⟨?_, ?_⟩
  · have hc := e_23_4
    simp only [e_22_11] at hc
    linear_combination c1004.trans k2 - hc
  · have hc := e_22_12
    simp only [e_22_10, e_23_5, e_22_11] at hc
    linear_combination c1005.trans k2 - hc

theorem privateBatchWrapper8_f142 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 23 27) = band (a (.virt 356)) (a (.virt 358)) := by
  have c1006 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1008 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_23_6 := arithEq_of_rows h (row := 23) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_6
  simp only [← c1006, ← c1007, ← c1008] at e_23_6
  have hr := e_23_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f143 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 23 31) = band (a (.virt 360)) (a (.virt 362)) := by
  have c1009 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1011 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_23_7 := arithEq_of_rows h (row := 23) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_7
  simp only [← c1009, ← c1010, ← c1011] at e_23_7
  have hr := e_23_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f144 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 23 35) = band (a (.wire 23 27)) (a (.wire 23 31)) := by
  have c1012 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_23_8 := arithEq_of_rows h (row := 23) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_8
  simp only [← c1012, ← c1013, ← c1014] at e_23_8
  have hr := e_23_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f145 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 31) = bor (a (.wire 4 15)) (a (.wire 23 35)) := by
  have c1015 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1017 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1019 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_8 := arithEq_of_rows h (row := 14) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c1015, ← c1016, ← c1017] at e_14_8
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c1018, ← c1019, k3, ← c1020] at e_16_7
  have hr := e_16_7
  simp only [e_14_8] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f146 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 31) = a (.virt 12) := by
  have c1021 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1021

theorem privateBatchWrapper8_f147 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 13) = a (.virt 0) := by
  have c1022 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1022

theorem privateBatchWrapper8_f148 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 29)) (a (.wire 20 43)) (a (.virt 364)) (a (.virt 365)) := by
  have c1023 := (privateBatchWrapper8_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1024 := (privateBatchWrapper8_copies32 a h).1
  have c1025 := (privateBatchWrapper8_copies32 a h).2.1
  have c1026 := (privateBatchWrapper8_copies32 a h).2.2.1
  have c1027 := (privateBatchWrapper8_copies32 a h).2.2.2.1
  have c1028 := (privateBatchWrapper8_copies32 a h).2.2.2.2.1
  have c1029 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.1
  have c1030 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_22_13 := arithEq_of_rows h (row := 22) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_13
  simp only [← c1023, k3, ← c1024, k3, ← c1025] at e_22_13
  have e_22_14 := arithEq_of_rows h (row := 22) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1026, ← c1027, k3, ← c1028] at e_22_14
  have e_23_9 := arithEq_of_rows h (row := 23) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_9
  simp only [← c1029, ← c1030, ← c1031] at e_23_9
  have e_23_10 := arithEq_of_rows h (row := 23) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_10
  simp only [← c1032, ← c1033, ← c1034] at e_23_10
  have e_24_0 := arithEq_of_rows h (row := 24) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_0
  simp only [← c1035, ← c1036, k3, ← c1037] at e_24_0
  refine ⟨?_, ?_⟩
  · have hc := e_23_9
    simp only [e_22_14] at hc
    linear_combination c1038.trans k2 - hc
  · have hc := e_24_0
    simp only [e_22_13, e_23_10, e_22_14] at hc
    linear_combination c1039.trans k2 - hc

theorem privateBatchWrapper8_f149 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 30)) (a (.wire 20 51)) (a (.virt 366)) (a (.virt 367)) := by
  have c1040 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1041 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (privateBatchWrapper8_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (privateBatchWrapper8_copies33 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_23_11 := arithEq_of_rows h (row := 23) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_11
  simp only [← c1046, ← c1047, ← c1048] at e_23_11
  have e_23_12 := arithEq_of_rows h (row := 23) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_12
  simp only [← c1049, ← c1050, ← c1051] at e_23_12
  have e_24_1 := arithEq_of_rows h (row := 24) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_1
  simp only [← c1040, k3, ← c1041, k3, ← c1042] at e_24_1
  have e_24_2 := arithEq_of_rows h (row := 24) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c1043, ← c1044, k3, ← c1045] at e_24_2
  have e_24_3 := arithEq_of_rows h (row := 24) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_3
  simp only [← c1052, ← c1053, k3, ← c1054] at e_24_3
  refine ⟨?_, ?_⟩
  · have hc := e_23_11
    simp only [e_24_2] at hc
    linear_combination c1055.trans k2 - hc
  · have hc := e_24_3
    simp only [e_24_1, e_23_12, e_24_2] at hc
    linear_combination c1056.trans k2 - hc

theorem privateBatchWrapper8_f150 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 31)) (a (.wire 20 59)) (a (.virt 368)) (a (.virt 369)) := by
  have c1057 := (privateBatchWrapper8_copies33 a h).2.1
  have c1058 := (privateBatchWrapper8_copies33 a h).2.2.1
  have c1059 := (privateBatchWrapper8_copies33 a h).2.2.2.1
  have c1060 := (privateBatchWrapper8_copies33 a h).2.2.2.2.1
  have c1061 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.1
  have c1062 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_23_13 := arithEq_of_rows h (row := 23) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_13
  simp only [← c1063, ← c1064, ← c1065] at e_23_13
  have e_23_14 := arithEq_of_rows h (row := 23) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_14
  simp only [← c1066, ← c1067, ← c1068] at e_23_14
  have e_24_4 := arithEq_of_rows h (row := 24) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_4
  simp only [← c1057, k3, ← c1058, k3, ← c1059] at e_24_4
  have e_24_5 := arithEq_of_rows h (row := 24) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c1060, ← c1061, k3, ← c1062] at e_24_5
  have e_24_6 := arithEq_of_rows h (row := 24) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_6
  simp only [← c1069, ← c1070, k3, ← c1071] at e_24_6
  refine ⟨?_, ?_⟩
  · have hc := e_23_13
    simp only [e_24_5] at hc
    linear_combination c1072.trans k2 - hc
  · have hc := e_24_6
    simp only [e_24_4, e_23_14, e_24_5] at hc
    linear_combination c1073.trans k2 - hc

theorem privateBatchWrapper8_f151 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 32)) (a (.wire 21 7)) (a (.virt 370)) (a (.virt 371)) := by
  have c1074 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1078 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (privateBatchWrapper8_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (privateBatchWrapper8_copies34 a h).1
  have c1089 := (privateBatchWrapper8_copies34 a h).2.1
  have c1090 := (privateBatchWrapper8_copies34 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_24_7 := arithEq_of_rows h (row := 24) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_7
  simp only [← c1074, k3, ← c1075, k3, ← c1076] at e_24_7
  have e_24_8 := arithEq_of_rows h (row := 24) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_8
  simp only [← c1077, ← c1078, k3, ← c1079] at e_24_8
  have e_24_9 := arithEq_of_rows h (row := 24) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_9
  simp only [← c1086, ← c1087, k3, ← c1088] at e_24_9
  have e_25_0 := arithEq_of_rows h (row := 25) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_0
  simp only [← c1080, ← c1081, ← c1082] at e_25_0
  have e_25_1 := arithEq_of_rows h (row := 25) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_1
  simp only [← c1083, ← c1084, ← c1085] at e_25_1
  refine ⟨?_, ?_⟩
  · have hc := e_25_0
    simp only [e_24_8] at hc
    linear_combination c1089.trans k2 - hc
  · have hc := e_24_9
    simp only [e_24_7, e_25_1, e_24_8] at hc
    linear_combination c1090.trans k2 - hc

theorem privateBatchWrapper8_f152 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 25 11) = band (a (.virt 364)) (a (.virt 366)) := by
  have c1091 := (privateBatchWrapper8_copies34 a h).2.2.2.1
  have c1092 := (privateBatchWrapper8_copies34 a h).2.2.2.2.1
  have c1093 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.1
  have e_25_2 := arithEq_of_rows h (row := 25) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_2
  simp only [← c1091, ← c1092, ← c1093] at e_25_2
  have hr := e_25_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f153 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 25 15) = band (a (.virt 368)) (a (.virt 370)) := by
  have c1094 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.1
  have c1096 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.1
  have e_25_3 := arithEq_of_rows h (row := 25) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_3
  simp only [← c1094, ← c1095, ← c1096] at e_25_3
  have hr := e_25_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f154 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 25 19) = band (a (.wire 25 11)) (a (.wire 25 15)) := by
  have c1097 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1099 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_25_4 := arithEq_of_rows h (row := 25) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_4
  simp only [← c1097, ← c1098, ← c1099] at e_25_4
  have hr := e_25_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f155 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 35) = bor (a (.wire 4 59)) (a (.wire 25 19)) := by
  have c1100 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1102 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1103 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_9 := arithEq_of_rows h (row := 14) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c1100, ← c1101, ← c1102] at e_14_9
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c1103, ← c1104, k3, ← c1105] at e_16_8
  have hr := e_16_8
  simp only [e_14_9] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f156 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 35) = a (.virt 12) := by
  have c1106 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1106

theorem privateBatchWrapper8_f157 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 24) = a (.virt 0) := by
  have c1107 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1107

theorem privateBatchWrapper8_f158 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 40)) (a (.wire 20 43)) (a (.virt 372)) (a (.virt 373)) := by
  have c1108 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (privateBatchWrapper8_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1120 := (privateBatchWrapper8_copies35 a h).1
  have c1121 := (privateBatchWrapper8_copies35 a h).2.1
  have c1122 := (privateBatchWrapper8_copies35 a h).2.2.1
  have c1123 := (privateBatchWrapper8_copies35 a h).2.2.2.1
  have c1124 := (privateBatchWrapper8_copies35 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_24_10 := arithEq_of_rows h (row := 24) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_10
  simp only [← c1108, k3, ← c1109, k3, ← c1110] at e_24_10
  have e_24_11 := arithEq_of_rows h (row := 24) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_11
  simp only [← c1111, ← c1112, k3, ← c1113] at e_24_11
  have e_24_12 := arithEq_of_rows h (row := 24) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_12
  simp only [← c1120, ← c1121, k3, ← c1122] at e_24_12
  have e_25_5 := arithEq_of_rows h (row := 25) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_5
  simp only [← c1114, ← c1115, ← c1116] at e_25_5
  have e_25_6 := arithEq_of_rows h (row := 25) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_6
  simp only [← c1117, ← c1118, ← c1119] at e_25_6
  refine ⟨?_, ?_⟩
  · have hc := e_25_5
    simp only [e_24_11] at hc
    linear_combination c1123.trans k2 - hc
  · have hc := e_24_12
    simp only [e_24_10, e_25_6, e_24_11] at hc
    linear_combination c1124.trans k2 - hc

theorem privateBatchWrapper8_f159 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 41)) (a (.wire 20 51)) (a (.virt 374)) (a (.virt 375)) := by
  have c1125 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.1
  have c1126 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1140 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_24_13 := arithEq_of_rows h (row := 24) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_13
  simp only [← c1125, k3, ← c1126, k3, ← c1127] at e_24_13
  have e_24_14 := arithEq_of_rows h (row := 24) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_14
  simp only [← c1128, ← c1129, k3, ← c1130] at e_24_14
  have e_25_7 := arithEq_of_rows h (row := 25) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_7
  simp only [← c1131, ← c1132, ← c1133] at e_25_7
  have e_25_8 := arithEq_of_rows h (row := 25) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_8
  simp only [← c1134, ← c1135, ← c1136] at e_25_8
  have e_26_0 := arithEq_of_rows h (row := 26) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1137, ← c1138, k3, ← c1139] at e_26_0
  refine ⟨?_, ?_⟩
  · have hc := e_25_7
    simp only [e_24_14] at hc
    linear_combination c1140.trans k2 - hc
  · have hc := e_26_0
    simp only [e_24_13, e_25_8, e_24_14] at hc
    linear_combination c1141.trans k2 - hc

theorem privateBatchWrapper8_f160 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 42)) (a (.wire 20 59)) (a (.virt 376)) (a (.virt 377)) := by
  have c1142 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1143 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1150 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (privateBatchWrapper8_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (privateBatchWrapper8_copies36 a h).1
  have c1153 := (privateBatchWrapper8_copies36 a h).2.1
  have c1154 := (privateBatchWrapper8_copies36 a h).2.2.1
  have c1155 := (privateBatchWrapper8_copies36 a h).2.2.2.1
  have c1156 := (privateBatchWrapper8_copies36 a h).2.2.2.2.1
  have c1157 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.1
  have c1158 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_25_9 := arithEq_of_rows h (row := 25) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_9
  simp only [← c1148, ← c1149, ← c1150] at e_25_9
  have e_25_10 := arithEq_of_rows h (row := 25) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_10
  simp only [← c1151, ← c1152, ← c1153] at e_25_10
  have e_26_1 := arithEq_of_rows h (row := 26) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1142, k3, ← c1143, k3, ← c1144] at e_26_1
  have e_26_2 := arithEq_of_rows h (row := 26) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1145, ← c1146, k3, ← c1147] at e_26_2
  have e_26_3 := arithEq_of_rows h (row := 26) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1154, ← c1155, k3, ← c1156] at e_26_3
  refine ⟨?_, ?_⟩
  · have hc := e_25_9
    simp only [e_26_2] at hc
    linear_combination c1157.trans k2 - hc
  · have hc := e_26_3
    simp only [e_26_1, e_25_10, e_26_2] at hc
    linear_combination c1158.trans k2 - hc

theorem privateBatchWrapper8_f161 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 43)) (a (.wire 21 7)) (a (.virt 378)) (a (.virt 379)) := by
  have c1159 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have c1162 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1174 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_25_11 := arithEq_of_rows h (row := 25) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_11
  simp only [← c1165, ← c1166, ← c1167] at e_25_11
  have e_25_12 := arithEq_of_rows h (row := 25) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_12
  simp only [← c1168, ← c1169, ← c1170] at e_25_12
  have e_26_4 := arithEq_of_rows h (row := 26) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1159, k3, ← c1160, k3, ← c1161] at e_26_4
  have e_26_5 := arithEq_of_rows h (row := 26) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1162, ← c1163, k3, ← c1164] at e_26_5
  have e_26_6 := arithEq_of_rows h (row := 26) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1171, ← c1172, k3, ← c1173] at e_26_6
  refine ⟨?_, ?_⟩
  · have hc := e_25_11
    simp only [e_26_5] at hc
    linear_combination c1174.trans k2 - hc
  · have hc := e_26_6
    simp only [e_26_4, e_25_12, e_26_5] at hc
    linear_combination c1175.trans k2 - hc

theorem privateBatchWrapper8_f162 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 25 55) = band (a (.virt 372)) (a (.virt 374)) := by
  have c1176 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1177 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_25_13 := arithEq_of_rows h (row := 25) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_13
  simp only [← c1176, ← c1177, ← c1178] at e_25_13
  have hr := e_25_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f163 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 25 59) = band (a (.virt 376)) (a (.virt 378)) := by
  have c1179 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1180 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_25_14 := arithEq_of_rows h (row := 25) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_14
  simp only [← c1179, ← c1180, ← c1181] at e_25_14
  have hr := e_25_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f164 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 27 3) = band (a (.wire 25 55)) (a (.wire 25 59)) := by
  have c1182 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1183 := (privateBatchWrapper8_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (privateBatchWrapper8_copies37 a h).1
  have e_27_0 := arithEq_of_rows h (row := 27) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1182, ← c1183, ← c1184] at e_27_0
  have hr := e_27_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f165 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 39) = bor (a (.wire 6 43)) (a (.wire 27 3)) := by
  have c1185 := (privateBatchWrapper8_copies37 a h).2.1
  have c1186 := (privateBatchWrapper8_copies37 a h).2.2.1
  have c1187 := (privateBatchWrapper8_copies37 a h).2.2.2.1
  have c1188 := (privateBatchWrapper8_copies37 a h).2.2.2.2.1
  have c1189 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.1
  have c1190 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_10 := arithEq_of_rows h (row := 14) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c1185, ← c1186, ← c1187] at e_14_10
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c1188, ← c1189, k3, ← c1190] at e_16_9
  have hr := e_16_9
  simp only [e_14_10] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f166 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 39) = a (.virt 12) := by
  have c1191 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.1
  exact c1191

theorem privateBatchWrapper8_f167 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 35) = a (.virt 0) := by
  have c1192 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.1
  exact c1192

theorem privateBatchWrapper8_f168 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 51)) (a (.wire 20 43)) (a (.virt 380)) (a (.virt 381)) := by
  have c1193 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1207 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_26_7 := arithEq_of_rows h (row := 26) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1193, k3, ← c1194, k3, ← c1195] at e_26_7
  have e_26_8 := arithEq_of_rows h (row := 26) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1196, ← c1197, k3, ← c1198] at e_26_8
  have e_26_9 := arithEq_of_rows h (row := 26) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1205, ← c1206, k3, ← c1207] at e_26_9
  have e_27_1 := arithEq_of_rows h (row := 27) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1199, ← c1200, ← c1201] at e_27_1
  have e_27_2 := arithEq_of_rows h (row := 27) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1202, ← c1203, ← c1204] at e_27_2
  refine ⟨?_, ?_⟩
  · have hc := e_27_1
    simp only [e_26_8] at hc
    linear_combination c1208.trans k2 - hc
  · have hc := e_26_9
    simp only [e_26_7, e_27_2, e_26_8] at hc
    linear_combination c1209.trans k2 - hc

theorem privateBatchWrapper8_f169 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 52)) (a (.wire 20 51)) (a (.virt 382)) (a (.virt 383)) := by
  have c1210 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (privateBatchWrapper8_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (privateBatchWrapper8_copies38 a h).1
  have c1217 := (privateBatchWrapper8_copies38 a h).2.1
  have c1218 := (privateBatchWrapper8_copies38 a h).2.2.1
  have c1219 := (privateBatchWrapper8_copies38 a h).2.2.2.1
  have c1220 := (privateBatchWrapper8_copies38 a h).2.2.2.2.1
  have c1221 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.1
  have c1222 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.1
  have c1224 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_26_10 := arithEq_of_rows h (row := 26) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1210, k3, ← c1211, k3, ← c1212] at e_26_10
  have e_26_11 := arithEq_of_rows h (row := 26) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1213, ← c1214, k3, ← c1215] at e_26_11
  have e_26_12 := arithEq_of_rows h (row := 26) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1222, ← c1223, k3, ← c1224] at e_26_12
  have e_27_3 := arithEq_of_rows h (row := 27) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1216, ← c1217, ← c1218] at e_27_3
  have e_27_4 := arithEq_of_rows h (row := 27) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_4
  simp only [← c1219, ← c1220, ← c1221] at e_27_4
  refine ⟨?_, ?_⟩
  · have hc := e_27_3
    simp only [e_26_11] at hc
    linear_combination c1225.trans k2 - hc
  · have hc := e_26_12
    simp only [e_26_10, e_27_4, e_26_11] at hc
    linear_combination c1226.trans k2 - hc

theorem privateBatchWrapper8_f170 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 53)) (a (.wire 20 59)) (a (.virt 384)) (a (.virt 385)) := by
  have c1227 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1233 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1236 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1239 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1241 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_26_13 := arithEq_of_rows h (row := 26) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1227, k3, ← c1228, k3, ← c1229] at e_26_13
  have e_26_14 := arithEq_of_rows h (row := 26) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1230, ← c1231, k3, ← c1232] at e_26_14
  have e_27_5 := arithEq_of_rows h (row := 27) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_5
  simp only [← c1233, ← c1234, ← c1235] at e_27_5
  have e_27_6 := arithEq_of_rows h (row := 27) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_6
  simp only [← c1236, ← c1237, ← c1238] at e_27_6
  have e_28_0 := arithEq_of_rows h (row := 28) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c1239, ← c1240, k3, ← c1241] at e_28_0
  refine ⟨?_, ?_⟩
  · have hc := e_27_5
    simp only [e_26_14] at hc
    linear_combination c1242.trans k2 - hc
  · have hc := e_28_0
    simp only [e_26_13, e_27_6, e_26_14] at hc
    linear_combination c1243.trans k2 - hc

theorem privateBatchWrapper8_f171 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 54)) (a (.wire 21 7)) (a (.virt 386)) (a (.virt 387)) := by
  have c1244 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1245 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (privateBatchWrapper8_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (privateBatchWrapper8_copies39 a h).1
  have c1249 := (privateBatchWrapper8_copies39 a h).2.1
  have c1250 := (privateBatchWrapper8_copies39 a h).2.2.1
  have c1251 := (privateBatchWrapper8_copies39 a h).2.2.2.1
  have c1252 := (privateBatchWrapper8_copies39 a h).2.2.2.2.1
  have c1253 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.1
  have c1254 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.1
  have c1256 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have c1258 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_27_7 := arithEq_of_rows h (row := 27) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_7
  simp only [← c1250, ← c1251, ← c1252] at e_27_7
  have e_27_8 := arithEq_of_rows h (row := 27) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_8
  simp only [← c1253, ← c1254, ← c1255] at e_27_8
  have e_28_1 := arithEq_of_rows h (row := 28) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c1244, k3, ← c1245, k3, ← c1246] at e_28_1
  have e_28_2 := arithEq_of_rows h (row := 28) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c1247, ← c1248, k3, ← c1249] at e_28_2
  have e_28_3 := arithEq_of_rows h (row := 28) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c1256, ← c1257, k3, ← c1258] at e_28_3
  refine ⟨?_, ?_⟩
  · have hc := e_27_7
    simp only [e_28_2] at hc
    linear_combination c1259.trans k2 - hc
  · have hc := e_28_3
    simp only [e_28_1, e_27_8, e_28_2] at hc
    linear_combination c1260.trans k2 - hc

theorem privateBatchWrapper8_f172 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 27 39) = band (a (.virt 380)) (a (.virt 382)) := by
  have c1261 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_9 := arithEq_of_rows h (row := 27) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_9
  simp only [← c1261, ← c1262, ← c1263] at e_27_9
  have hr := e_27_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f173 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 27 43) = band (a (.virt 384)) (a (.virt 386)) := by
  have c1264 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_10 := arithEq_of_rows h (row := 27) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_10
  simp only [← c1264, ← c1265, ← c1266] at e_27_10
  have hr := e_27_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f174 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 27 47) = band (a (.wire 27 39)) (a (.wire 27 43)) := by
  have c1267 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_11 := arithEq_of_rows h (row := 27) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_11
  simp only [← c1267, ← c1268, ← c1269] at e_27_11
  have hr := e_27_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f175 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 43) = bor (a (.wire 8 27)) (a (.wire 27 47)) := by
  have c1270 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1271 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1274 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_11 := arithEq_of_rows h (row := 14) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c1270, ← c1271, ← c1272] at e_14_11
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c1273, ← c1274, k3, ← c1275] at e_16_10
  have hr := e_16_10
  simp only [e_14_11] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f176 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 43) = a (.virt 12) := by
  have c1276 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1276

theorem privateBatchWrapper8_f177 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 46) = a (.virt 0) := by
  have c1277 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1277

theorem privateBatchWrapper8_f178 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 62)) (a (.wire 20 43)) (a (.virt 388)) (a (.virt 389)) := by
  have c1278 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (privateBatchWrapper8_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (privateBatchWrapper8_copies40 a h).1
  have c1281 := (privateBatchWrapper8_copies40 a h).2.1
  have c1282 := (privateBatchWrapper8_copies40 a h).2.2.1
  have c1283 := (privateBatchWrapper8_copies40 a h).2.2.2.1
  have c1284 := (privateBatchWrapper8_copies40 a h).2.2.2.2.1
  have c1285 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.1
  have c1286 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.1
  have c1287 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1291 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_27_12 := arithEq_of_rows h (row := 27) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_12
  simp only [← c1284, ← c1285, ← c1286] at e_27_12
  have e_27_13 := arithEq_of_rows h (row := 27) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_13
  simp only [← c1287, ← c1288, ← c1289] at e_27_13
  have e_28_4 := arithEq_of_rows h (row := 28) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c1278, k3, ← c1279, k3, ← c1280] at e_28_4
  have e_28_5 := arithEq_of_rows h (row := 28) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1281, ← c1282, k3, ← c1283] at e_28_5
  have e_28_6 := arithEq_of_rows h (row := 28) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1290, ← c1291, k3, ← c1292] at e_28_6
  refine ⟨?_, ?_⟩
  · have hc := e_27_12
    simp only [e_28_5] at hc
    linear_combination c1293.trans k2 - hc
  · have hc := e_28_6
    simp only [e_28_4, e_27_13, e_28_5] at hc
    linear_combination c1294.trans k2 - hc

theorem privateBatchWrapper8_f179 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 63)) (a (.wire 20 51)) (a (.virt 390)) (a (.virt 391)) := by
  have c1295 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1308 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (privateBatchWrapper8_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_27_14 := arithEq_of_rows h (row := 27) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_14
  simp only [← c1301, ← c1302, ← c1303] at e_27_14
  have e_28_7 := arithEq_of_rows h (row := 28) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1295, k3, ← c1296, k3, ← c1297] at e_28_7
  have e_28_8 := arithEq_of_rows h (row := 28) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1298, ← c1299, k3, ← c1300] at e_28_8
  have e_28_9 := arithEq_of_rows h (row := 28) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1307, ← c1308, k3, ← c1309] at e_28_9
  have e_29_0 := arithEq_of_rows h (row := 29) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1304, ← c1305, ← c1306] at e_29_0
  refine ⟨?_, ?_⟩
  · have hc := e_27_14
    simp only [e_28_8] at hc
    linear_combination c1310.trans k2 - hc
  · have hc := e_28_9
    simp only [e_28_7, e_29_0, e_28_8] at hc
    linear_combination c1311.trans k2 - hc

theorem privateBatchWrapper8_f180 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 64)) (a (.wire 20 59)) (a (.virt 392)) (a (.virt 393)) := by
  have c1312 := (privateBatchWrapper8_copies41 a h).1
  have c1313 := (privateBatchWrapper8_copies41 a h).2.1
  have c1314 := (privateBatchWrapper8_copies41 a h).2.2.1
  have c1315 := (privateBatchWrapper8_copies41 a h).2.2.2.1
  have c1316 := (privateBatchWrapper8_copies41 a h).2.2.2.2.1
  have c1317 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.1
  have c1318 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.1
  have c1319 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.1
  have c1321 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have c1322 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1325 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_28_10 := arithEq_of_rows h (row := 28) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1312, k3, ← c1313, k3, ← c1314] at e_28_10
  have e_28_11 := arithEq_of_rows h (row := 28) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1315, ← c1316, k3, ← c1317] at e_28_11
  have e_28_12 := arithEq_of_rows h (row := 28) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1324, ← c1325, k3, ← c1326] at e_28_12
  have e_29_1 := arithEq_of_rows h (row := 29) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1318, ← c1319, ← c1320] at e_29_1
  have e_29_2 := arithEq_of_rows h (row := 29) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1321, ← c1322, ← c1323] at e_29_2
  refine ⟨?_, ?_⟩
  · have hc := e_29_1
    simp only [e_28_11] at hc
    linear_combination c1327.trans k2 - hc
  · have hc := e_28_12
    simp only [e_28_10, e_29_2, e_28_11] at hc
    linear_combination c1328.trans k2 - hc

theorem privateBatchWrapper8_f181 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 65)) (a (.wire 21 7)) (a (.virt 394)) (a (.virt 395)) := by
  have c1329 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1335 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1342 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (privateBatchWrapper8_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (privateBatchWrapper8_copies42 a h).1
  have c1345 := (privateBatchWrapper8_copies42 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_28_13 := arithEq_of_rows h (row := 28) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1329, k3, ← c1330, k3, ← c1331] at e_28_13
  have e_28_14 := arithEq_of_rows h (row := 28) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1332, ← c1333, k3, ← c1334] at e_28_14
  have e_29_3 := arithEq_of_rows h (row := 29) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1335, ← c1336, ← c1337] at e_29_3
  have e_29_4 := arithEq_of_rows h (row := 29) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1338, ← c1339, ← c1340] at e_29_4
  have e_30_0 := arithEq_of_rows h (row := 30) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c1341, ← c1342, k3, ← c1343] at e_30_0
  refine ⟨?_, ?_⟩
  · have hc := e_29_3
    simp only [e_28_14] at hc
    linear_combination c1344.trans k2 - hc
  · have hc := e_30_0
    simp only [e_28_13, e_29_4, e_28_14] at hc
    linear_combination c1345.trans k2 - hc

theorem privateBatchWrapper8_f182 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 29 23) = band (a (.virt 388)) (a (.virt 390)) := by
  have c1346 := (privateBatchWrapper8_copies42 a h).2.2.1
  have c1347 := (privateBatchWrapper8_copies42 a h).2.2.2.1
  have c1348 := (privateBatchWrapper8_copies42 a h).2.2.2.2.1
  have e_29_5 := arithEq_of_rows h (row := 29) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1346, ← c1347, ← c1348] at e_29_5
  have hr := e_29_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f183 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 29 27) = band (a (.virt 392)) (a (.virt 394)) := by
  have c1349 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.1
  have c1350 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.1
  have c1351 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.1
  have e_29_6 := arithEq_of_rows h (row := 29) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1349, ← c1350, ← c1351] at e_29_6
  have hr := e_29_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f184 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 29 31) = band (a (.wire 29 23)) (a (.wire 29 27)) := by
  have c1352 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have c1354 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_29_7 := arithEq_of_rows h (row := 29) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c1352, ← c1353, ← c1354] at e_29_7
  have hr := e_29_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f185 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 47) = bor (a (.wire 10 11)) (a (.wire 29 31)) := by
  have c1355 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1358 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_12 := arithEq_of_rows h (row := 14) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c1355, ← c1356, ← c1357] at e_14_12
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1358, ← c1359, k3, ← c1360] at e_16_11
  have hr := e_16_11
  simp only [e_14_12] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f186 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 47) = a (.virt 12) := by
  have c1361 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1361

theorem privateBatchWrapper8_f187 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 57) = a (.virt 0) := by
  have c1362 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1362

theorem privateBatchWrapper8_f188 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 73)) (a (.wire 20 43)) (a (.virt 396)) (a (.virt 397)) := by
  have c1363 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1375 := (privateBatchWrapper8_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (privateBatchWrapper8_copies43 a h).1
  have c1377 := (privateBatchWrapper8_copies43 a h).2.1
  have c1378 := (privateBatchWrapper8_copies43 a h).2.2.1
  have c1379 := (privateBatchWrapper8_copies43 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_29_8 := arithEq_of_rows h (row := 29) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c1369, ← c1370, ← c1371] at e_29_8
  have e_29_9 := arithEq_of_rows h (row := 29) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c1372, ← c1373, ← c1374] at e_29_9
  have e_30_1 := arithEq_of_rows h (row := 30) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c1363, k3, ← c1364, k3, ← c1365] at e_30_1
  have e_30_2 := arithEq_of_rows h (row := 30) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c1366, ← c1367, k3, ← c1368] at e_30_2
  have e_30_3 := arithEq_of_rows h (row := 30) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c1375, ← c1376, k3, ← c1377] at e_30_3
  refine ⟨?_, ?_⟩
  · have hc := e_29_8
    simp only [e_30_2] at hc
    linear_combination c1378.trans k2 - hc
  · have hc := e_30_3
    simp only [e_30_1, e_29_9, e_30_2] at hc
    linear_combination c1379.trans k2 - hc

theorem privateBatchWrapper8_f189 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 74)) (a (.wire 20 51)) (a (.virt 398)) (a (.virt 399)) := by
  have c1380 := (privateBatchWrapper8_copies43 a h).2.2.2.2.1
  have c1381 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.1
  have c1382 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.1
  have c1383 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have c1386 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1389 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1392 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_29_10 := arithEq_of_rows h (row := 29) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c1386, ← c1387, ← c1388] at e_29_10
  have e_29_11 := arithEq_of_rows h (row := 29) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c1389, ← c1390, ← c1391] at e_29_11
  have e_30_4 := arithEq_of_rows h (row := 30) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c1380, k3, ← c1381, k3, ← c1382] at e_30_4
  have e_30_5 := arithEq_of_rows h (row := 30) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c1383, ← c1384, k3, ← c1385] at e_30_5
  have e_30_6 := arithEq_of_rows h (row := 30) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c1392, ← c1393, k3, ← c1394] at e_30_6
  refine ⟨?_, ?_⟩
  · have hc := e_29_10
    simp only [e_30_5] at hc
    linear_combination c1395.trans k2 - hc
  · have hc := e_30_6
    simp only [e_30_4, e_29_11, e_30_5] at hc
    linear_combination c1396.trans k2 - hc

theorem privateBatchWrapper8_f190 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 75)) (a (.wire 20 59)) (a (.virt 400)) (a (.virt 401)) := by
  have c1397 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (privateBatchWrapper8_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (privateBatchWrapper8_copies44 a h).1
  have c1409 := (privateBatchWrapper8_copies44 a h).2.1
  have c1410 := (privateBatchWrapper8_copies44 a h).2.2.1
  have c1411 := (privateBatchWrapper8_copies44 a h).2.2.2.1
  have c1412 := (privateBatchWrapper8_copies44 a h).2.2.2.2.1
  have c1413 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_29_12 := arithEq_of_rows h (row := 29) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c1403, ← c1404, ← c1405] at e_29_12
  have e_29_13 := arithEq_of_rows h (row := 29) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c1406, ← c1407, ← c1408] at e_29_13
  have e_30_7 := arithEq_of_rows h (row := 30) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c1397, k3, ← c1398, k3, ← c1399] at e_30_7
  have e_30_8 := arithEq_of_rows h (row := 30) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c1400, ← c1401, k3, ← c1402] at e_30_8
  have e_30_9 := arithEq_of_rows h (row := 30) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c1409, ← c1410, k3, ← c1411] at e_30_9
  refine ⟨?_, ?_⟩
  · have hc := e_29_12
    simp only [e_30_8] at hc
    linear_combination c1412.trans k2 - hc
  · have hc := e_30_9
    simp only [e_30_7, e_29_13, e_30_8] at hc
    linear_combination c1413.trans k2 - hc

theorem privateBatchWrapper8_f191 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 76)) (a (.wire 21 7)) (a (.virt 402)) (a (.virt 403)) := by
  have c1414 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.1
  have c1417 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1423 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1425 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1426 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1428 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1429 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_29_14 := arithEq_of_rows h (row := 29) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c1420, ← c1421, ← c1422] at e_29_14
  have e_30_10 := arithEq_of_rows h (row := 30) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c1414, k3, ← c1415, k3, ← c1416] at e_30_10
  have e_30_11 := arithEq_of_rows h (row := 30) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c1417, ← c1418, k3, ← c1419] at e_30_11
  have e_30_12 := arithEq_of_rows h (row := 30) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c1426, ← c1427, k3, ← c1428] at e_30_12
  have e_31_0 := arithEq_of_rows h (row := 31) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c1423, ← c1424, ← c1425] at e_31_0
  refine ⟨?_, ?_⟩
  · have hc := e_29_14
    simp only [e_30_11] at hc
    linear_combination c1429.trans k2 - hc
  · have hc := e_30_12
    simp only [e_30_10, e_31_0, e_30_11] at hc
    linear_combination c1430.trans k2 - hc

theorem privateBatchWrapper8_f192 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 31 7) = band (a (.virt 396)) (a (.virt 398)) := by
  have c1431 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1432 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_1 := arithEq_of_rows h (row := 31) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c1431, ← c1432, ← c1433] at e_31_1
  have hr := e_31_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f193 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 31 11) = band (a (.virt 400)) (a (.virt 402)) := by
  have c1434 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1435 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_2 := arithEq_of_rows h (row := 31) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c1434, ← c1435, ← c1436] at e_31_2
  have hr := e_31_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f194 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 31 15) = band (a (.wire 31 7)) (a (.wire 31 11)) := by
  have c1437 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1438 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1439 := (privateBatchWrapper8_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_31_3 := arithEq_of_rows h (row := 31) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c1437, ← c1438, ← c1439] at e_31_3
  have hr := e_31_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f195 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 51) = bor (a (.wire 10 55)) (a (.wire 31 15)) := by
  have c1440 := (privateBatchWrapper8_copies45 a h).1
  have c1441 := (privateBatchWrapper8_copies45 a h).2.1
  have c1442 := (privateBatchWrapper8_copies45 a h).2.2.1
  have c1443 := (privateBatchWrapper8_copies45 a h).2.2.2.1
  have c1444 := (privateBatchWrapper8_copies45 a h).2.2.2.2.1
  have c1445 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_13 := arithEq_of_rows h (row := 14) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c1440, ← c1441, ← c1442] at e_14_13
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1443, ← c1444, k3, ← c1445] at e_16_12
  have hr := e_16_12
  simp only [e_14_13] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f196 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 51) = a (.virt 12) := by
  have c1446 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.1
  exact c1446

theorem privateBatchWrapper8_f197 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 68) = a (.virt 0) := by
  have c1447 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.1
  exact c1447

theorem privateBatchWrapper8_f198 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 84)) (a (.wire 20 43)) (a (.virt 404)) (a (.virt 405)) := by
  have c1448 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1451 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1454 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1456 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1459 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_30_13 := arithEq_of_rows h (row := 30) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c1448, k3, ← c1449, k3, ← c1450] at e_30_13
  have e_30_14 := arithEq_of_rows h (row := 30) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c1451, ← c1452, k3, ← c1453] at e_30_14
  have e_31_4 := arithEq_of_rows h (row := 31) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c1454, ← c1455, ← c1456] at e_31_4
  have e_31_5 := arithEq_of_rows h (row := 31) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c1457, ← c1458, ← c1459] at e_31_5
  have e_32_0 := arithEq_of_rows h (row := 32) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_0
  simp only [← c1460, ← c1461, k3, ← c1462] at e_32_0
  refine ⟨?_, ?_⟩
  · have hc := e_31_4
    simp only [e_30_14] at hc
    linear_combination c1463.trans k2 - hc
  · have hc := e_32_0
    simp only [e_30_13, e_31_5, e_30_14] at hc
    linear_combination c1464.trans k2 - hc

theorem privateBatchWrapper8_f199 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 85)) (a (.wire 20 51)) (a (.virt 406)) (a (.virt 407)) := by
  have c1465 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1468 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1469 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (privateBatchWrapper8_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (privateBatchWrapper8_copies46 a h).1
  have c1473 := (privateBatchWrapper8_copies46 a h).2.1
  have c1474 := (privateBatchWrapper8_copies46 a h).2.2.1
  have c1475 := (privateBatchWrapper8_copies46 a h).2.2.2.1
  have c1476 := (privateBatchWrapper8_copies46 a h).2.2.2.2.1
  have c1477 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.1
  have c1478 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.1
  have c1479 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_31_6 := arithEq_of_rows h (row := 31) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c1471, ← c1472, ← c1473] at e_31_6
  have e_31_7 := arithEq_of_rows h (row := 31) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c1474, ← c1475, ← c1476] at e_31_7
  have e_32_1 := arithEq_of_rows h (row := 32) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_1
  simp only [← c1465, k3, ← c1466, k3, ← c1467] at e_32_1
  have e_32_2 := arithEq_of_rows h (row := 32) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_2
  simp only [← c1468, ← c1469, k3, ← c1470] at e_32_2
  have e_32_3 := arithEq_of_rows h (row := 32) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_3
  simp only [← c1477, ← c1478, k3, ← c1479] at e_32_3
  refine ⟨?_, ?_⟩
  · have hc := e_31_6
    simp only [e_32_2] at hc
    linear_combination c1480.trans k2 - hc
  · have hc := e_32_3
    simp only [e_32_1, e_31_7, e_32_2] at hc
    linear_combination c1481.trans k2 - hc

theorem privateBatchWrapper8_f200 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 86)) (a (.wire 20 59)) (a (.virt 408)) (a (.virt 409)) := by
  have c1482 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1486 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1493 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1496 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_31_8 := arithEq_of_rows h (row := 31) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c1488, ← c1489, ← c1490] at e_31_8
  have e_31_9 := arithEq_of_rows h (row := 31) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c1491, ← c1492, ← c1493] at e_31_9
  have e_32_4 := arithEq_of_rows h (row := 32) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_4
  simp only [← c1482, k3, ← c1483, k3, ← c1484] at e_32_4
  have e_32_5 := arithEq_of_rows h (row := 32) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_5
  simp only [← c1485, ← c1486, k3, ← c1487] at e_32_5
  have e_32_6 := arithEq_of_rows h (row := 32) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_6
  simp only [← c1494, ← c1495, k3, ← c1496] at e_32_6
  refine ⟨?_, ?_⟩
  · have hc := e_31_8
    simp only [e_32_5] at hc
    linear_combination c1497.trans k2 - hc
  · have hc := e_32_6
    simp only [e_32_4, e_31_9, e_32_5] at hc
    linear_combination c1498.trans k2 - hc

theorem privateBatchWrapper8_f201 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 87)) (a (.wire 21 7)) (a (.virt 410)) (a (.virt 411)) := by
  have c1499 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1502 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1503 := (privateBatchWrapper8_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (privateBatchWrapper8_copies47 a h).1
  have c1505 := (privateBatchWrapper8_copies47 a h).2.1
  have c1506 := (privateBatchWrapper8_copies47 a h).2.2.1
  have c1507 := (privateBatchWrapper8_copies47 a h).2.2.2.1
  have c1508 := (privateBatchWrapper8_copies47 a h).2.2.2.2.1
  have c1509 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.1
  have c1510 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.1
  have c1513 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have c1514 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_31_10 := arithEq_of_rows h (row := 31) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c1505, ← c1506, ← c1507] at e_31_10
  have e_31_11 := arithEq_of_rows h (row := 31) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c1508, ← c1509, ← c1510] at e_31_11
  have e_32_7 := arithEq_of_rows h (row := 32) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_7
  simp only [← c1499, k3, ← c1500, k3, ← c1501] at e_32_7
  have e_32_8 := arithEq_of_rows h (row := 32) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_8
  simp only [← c1502, ← c1503, k3, ← c1504] at e_32_8
  have e_32_9 := arithEq_of_rows h (row := 32) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_9
  simp only [← c1511, ← c1512, k3, ← c1513] at e_32_9
  refine ⟨?_, ?_⟩
  · have hc := e_31_10
    simp only [e_32_8] at hc
    linear_combination c1514.trans k2 - hc
  · have hc := e_32_9
    simp only [e_32_7, e_31_11, e_32_8] at hc
    linear_combination c1515.trans k2 - hc

theorem privateBatchWrapper8_f202 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 31 51) = band (a (.virt 404)) (a (.virt 406)) := by
  have c1516 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_12 := arithEq_of_rows h (row := 31) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c1516, ← c1517, ← c1518] at e_31_12
  have hr := e_31_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f203 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 31 55) = band (a (.virt 408)) (a (.virt 410)) := by
  have c1519 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1520 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_13 := arithEq_of_rows h (row := 31) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c1519, ← c1520, ← c1521] at e_31_13
  have hr := e_31_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f204 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 31 59) = band (a (.wire 31 51)) (a (.wire 31 55)) := by
  have c1522 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1523 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_14 := arithEq_of_rows h (row := 31) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c1522, ← c1523, ← c1524] at e_31_14
  have hr := e_31_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f205 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 55) = bor (a (.wire 11 39)) (a (.wire 31 59)) := by
  have c1525 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1526 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1530 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_14 := arithEq_of_rows h (row := 14) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c1525, ← c1526, ← c1527] at e_14_14
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1528, ← c1529, k3, ← c1530] at e_16_13
  have hr := e_16_13
  simp only [e_14_14] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper8_f206 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 55) = a (.virt 12) := by
  have c1531 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1531

theorem privateBatchWrapper8_f207 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 79) = a (.virt 0) := by
  have c1532 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1532

theorem privateBatchWrapper8_f208 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 32 47) = bselect (a (.wire 3 31)) (a (.virt 11)) (a (.virt 10)) := by
  have c1533 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1535 := (privateBatchWrapper8_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (privateBatchWrapper8_copies48 a h).1
  have c1537 := (privateBatchWrapper8_copies48 a h).2.1
  have c1538 := (privateBatchWrapper8_copies48 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_10 := arithEq_of_rows h (row := 32) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_10
  simp only [← c1533, ← c1534, ← c1535] at e_32_10
  have e_32_11 := arithEq_of_rows h (row := 32) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_11
  simp only [← c1536, ← c1537, k2, ← c1538] at e_32_11
  have hr := e_32_11
  simp only [e_32_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f209 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 32 47) = a (.virt 11) + a (.wire 32 47) := by
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [k2]
  ring

theorem privateBatchWrapper8_f210 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 32 55) = bselect (a (.wire 4 15)) (a (.virt 11)) (a (.virt 23)) := by
  have c1539 := (privateBatchWrapper8_copies48 a h).2.2.2.1
  have c1540 := (privateBatchWrapper8_copies48 a h).2.2.2.2.1
  have c1541 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.1
  have c1542 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.1
  have c1543 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.1
  have c1544 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_12 := arithEq_of_rows h (row := 32) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_12
  simp only [← c1539, ← c1540, ← c1541] at e_32_12
  have e_32_13 := arithEq_of_rows h (row := 32) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_13
  simp only [← c1542, ← c1543, k2, ← c1544] at e_32_13
  have hr := e_32_13
  simp only [e_32_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f211 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 16 59) = a (.wire 32 47) + a (.wire 32 55) := by
  have c1545 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have c1546 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1547 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1545, ← c1546, k3, ← c1547] at e_16_14
  have hr := e_16_14
  linear_combination hr

theorem privateBatchWrapper8_f212 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 3) = bselect (a (.wire 4 59)) (a (.virt 11)) (a (.virt 34)) := by
  have c1548 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1553 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_14 := arithEq_of_rows h (row := 32) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_14
  simp only [← c1548, ← c1549, ← c1550] at e_32_14
  have e_33_0 := arithEq_of_rows h (row := 33) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c1551, ← c1552, k2, ← c1553] at e_33_0
  have hr := e_33_0
  simp only [e_32_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f213 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 3) = a (.wire 16 59) + a (.wire 33 3) := by
  have c1554 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1556 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_0 := arithEq_of_rows h (row := 34) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c1554, ← c1555, k3, ← c1556] at e_34_0
  have hr := e_34_0
  linear_combination hr

theorem privateBatchWrapper8_f214 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 11) = bselect (a (.wire 6 43)) (a (.virt 11)) (a (.virt 45)) := by
  have c1557 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1558 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1559 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1561 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1562 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_1 := arithEq_of_rows h (row := 33) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c1557, ← c1558, ← c1559] at e_33_1
  have e_33_2 := arithEq_of_rows h (row := 33) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c1560, ← c1561, k2, ← c1562] at e_33_2
  have hr := e_33_2
  simp only [e_33_1] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f215 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 7) = a (.wire 34 3) + a (.wire 33 11) := by
  have c1563 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_1 := arithEq_of_rows h (row := 34) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_1
  simp only [← c1563, ← c1564, k3, ← c1565] at e_34_1
  have hr := e_34_1
  linear_combination hr

theorem privateBatchWrapper8_f216 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 19) = bselect (a (.wire 8 27)) (a (.virt 11)) (a (.virt 56)) := by
  have c1566 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (privateBatchWrapper8_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (privateBatchWrapper8_copies49 a h).1
  have c1569 := (privateBatchWrapper8_copies49 a h).2.1
  have c1570 := (privateBatchWrapper8_copies49 a h).2.2.1
  have c1571 := (privateBatchWrapper8_copies49 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_3 := arithEq_of_rows h (row := 33) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c1566, ← c1567, ← c1568] at e_33_3
  have e_33_4 := arithEq_of_rows h (row := 33) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c1569, ← c1570, k2, ← c1571] at e_33_4
  have hr := e_33_4
  simp only [e_33_3] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f217 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 11) = a (.wire 34 7) + a (.wire 33 19) := by
  have c1572 := (privateBatchWrapper8_copies49 a h).2.2.2.2.1
  have c1573 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.1
  have c1574 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_2 := arithEq_of_rows h (row := 34) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_2
  simp only [← c1572, ← c1573, k3, ← c1574] at e_34_2
  have hr := e_34_2
  linear_combination hr

theorem privateBatchWrapper8_f218 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 27) = bselect (a (.wire 10 11)) (a (.virt 11)) (a (.virt 67)) := by
  have c1575 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have c1578 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1580 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_5 := arithEq_of_rows h (row := 33) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c1575, ← c1576, ← c1577] at e_33_5
  have e_33_6 := arithEq_of_rows h (row := 33) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c1578, ← c1579, k2, ← c1580] at e_33_6
  have hr := e_33_6
  simp only [e_33_5] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f219 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 15) = a (.wire 34 11) + a (.wire 33 27) := by
  have c1581 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1583 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_3 := arithEq_of_rows h (row := 34) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c1581, ← c1582, k3, ← c1583] at e_34_3
  have hr := e_34_3
  linear_combination hr

theorem privateBatchWrapper8_f220 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 35) = bselect (a (.wire 10 55)) (a (.virt 11)) (a (.virt 78)) := by
  have c1584 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1586 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_7 := arithEq_of_rows h (row := 33) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1584, ← c1585, ← c1586] at e_33_7
  have e_33_8 := arithEq_of_rows h (row := 33) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c1587, ← c1588, k2, ← c1589] at e_33_8
  have hr := e_33_8
  simp only [e_33_7] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f221 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 19) = a (.wire 34 15) + a (.wire 33 35) := by
  have c1590 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1592 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_4 := arithEq_of_rows h (row := 34) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_4
  simp only [← c1590, ← c1591, k3, ← c1592] at e_34_4
  have hr := e_34_4
  linear_combination hr

theorem privateBatchWrapper8_f222 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 43) = bselect (a (.wire 11 39)) (a (.virt 11)) (a (.virt 89)) := by
  have c1593 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1595 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1597 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1598 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_9 := arithEq_of_rows h (row := 33) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c1593, ← c1594, ← c1595] at e_33_9
  have e_33_10 := arithEq_of_rows h (row := 33) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c1596, ← c1597, k2, ← c1598] at e_33_10
  have hr := e_33_10
  simp only [e_33_9] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem privateBatchWrapper8_f223 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 23) = a (.wire 34 19) + a (.wire 33 43) := by
  have c1599 := (privateBatchWrapper8_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (privateBatchWrapper8_copies50 a h).1
  have c1601 := (privateBatchWrapper8_copies50 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_5 := arithEq_of_rows h (row := 34) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_5
  simp only [← c1599, ← c1600, k3, ← c1601] at e_34_5
  have hr := e_34_5
  linear_combination hr

theorem privateBatchWrapper8_f224 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 95)) 32 := by
  have c1602 := (privateBatchWrapper8_copies50 a h).2.2.1
  have c1603 := (privateBatchWrapper8_copies50 a h).2.2.2.1
  have c1604 := (privateBatchWrapper8_copies50 a h).2.2.2.2.1
  have c1605 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.1
  have c1606 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.1
  have c1607 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.1
  have c1609 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have c1610 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1612 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1613 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1614 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1616 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1617 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1619 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1620 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1622 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1623 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1626 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1628 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1629 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 35) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1602.trans k2
    · exact c1603.trans k2
    · exact c1604.trans k2
    · exact c1605.trans k2
    · exact c1606.trans k2
    · exact c1607.trans k2
    · exact c1608.trans k2
    · exact c1609.trans k2
    · exact c1610.trans k2
    · exact c1611.trans k2
    · exact c1612.trans k2
    · exact c1613.trans k2
    · exact c1614.trans k2
    · exact c1615.trans k2
    · exact c1616.trans k2
    · exact c1617.trans k2
    · exact c1618.trans k2
    · exact c1619.trans k2
    · exact c1620.trans k2
    · exact c1621.trans k2
    · exact c1622.trans k2
    · exact c1623.trans k2
    · exact c1624.trans k2
    · exact c1625.trans k2
    · exact c1626.trans k2
    · exact c1627.trans k2
    · exact c1628.trans k2
    )
  rwa [c1629] at hr

theorem privateBatchWrapper8_f225 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.virt 95) = a (.virt 11) + a (.virt 95) := by
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [k2]
  ring

theorem privateBatchWrapper8_f226 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 100)) 32 := by
  have c1630 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1631 := (privateBatchWrapper8_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (privateBatchWrapper8_copies51 a h).1
  have c1633 := (privateBatchWrapper8_copies51 a h).2.1
  have c1634 := (privateBatchWrapper8_copies51 a h).2.2.1
  have c1635 := (privateBatchWrapper8_copies51 a h).2.2.2.1
  have c1636 := (privateBatchWrapper8_copies51 a h).2.2.2.2.1
  have c1637 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.1
  have c1638 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.1
  have c1640 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1643 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1646 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1649 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1655 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1657 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 36) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1630.trans k2
    · exact c1631.trans k2
    · exact c1632.trans k2
    · exact c1633.trans k2
    · exact c1634.trans k2
    · exact c1635.trans k2
    · exact c1636.trans k2
    · exact c1637.trans k2
    · exact c1638.trans k2
    · exact c1639.trans k2
    · exact c1640.trans k2
    · exact c1641.trans k2
    · exact c1642.trans k2
    · exact c1643.trans k2
    · exact c1644.trans k2
    · exact c1645.trans k2
    · exact c1646.trans k2
    · exact c1647.trans k2
    · exact c1648.trans k2
    · exact c1649.trans k2
    · exact c1650.trans k2
    · exact c1651.trans k2
    · exact c1652.trans k2
    · exact c1653.trans k2
    · exact c1654.trans k2
    · exact c1655.trans k2
    · exact c1656.trans k2
    )
  rwa [c1657] at hr

theorem privateBatchWrapper8_f227 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 27) = a (.virt 95) + a (.virt 100) := by
  have c1658 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1660 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_6 := arithEq_of_rows h (row := 34) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_6
  simp only [← c1658, ← c1659, k3, ← c1660] at e_34_6
  have hr := e_34_6
  linear_combination hr

theorem privateBatchWrapper8_f228 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 105)) 32 := by
  have c1661 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1663 := (privateBatchWrapper8_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (privateBatchWrapper8_copies52 a h).1
  have c1665 := (privateBatchWrapper8_copies52 a h).2.1
  have c1666 := (privateBatchWrapper8_copies52 a h).2.2.1
  have c1667 := (privateBatchWrapper8_copies52 a h).2.2.2.1
  have c1668 := (privateBatchWrapper8_copies52 a h).2.2.2.2.1
  have c1669 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.1
  have c1670 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.1
  have c1671 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.1
  have c1674 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1677 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1679 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1685 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1688 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 37) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1661.trans k2
    · exact c1662.trans k2
    · exact c1663.trans k2
    · exact c1664.trans k2
    · exact c1665.trans k2
    · exact c1666.trans k2
    · exact c1667.trans k2
    · exact c1668.trans k2
    · exact c1669.trans k2
    · exact c1670.trans k2
    · exact c1671.trans k2
    · exact c1672.trans k2
    · exact c1673.trans k2
    · exact c1674.trans k2
    · exact c1675.trans k2
    · exact c1676.trans k2
    · exact c1677.trans k2
    · exact c1678.trans k2
    · exact c1679.trans k2
    · exact c1680.trans k2
    · exact c1681.trans k2
    · exact c1682.trans k2
    · exact c1683.trans k2
    · exact c1684.trans k2
    · exact c1685.trans k2
    · exact c1686.trans k2
    · exact c1687.trans k2
    )
  rwa [c1688] at hr

theorem privateBatchWrapper8_f229 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 31) = a (.wire 34 27) + a (.virt 105) := by
  have c1689 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1691 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_7 := arithEq_of_rows h (row := 34) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_7
  simp only [← c1689, ← c1690, k3, ← c1691] at e_34_7
  have hr := e_34_7
  linear_combination hr

theorem privateBatchWrapper8_f230 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 110)) 32 := by
  have c1692 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1694 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (privateBatchWrapper8_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (privateBatchWrapper8_copies53 a h).1
  have c1697 := (privateBatchWrapper8_copies53 a h).2.1
  have c1698 := (privateBatchWrapper8_copies53 a h).2.2.1
  have c1699 := (privateBatchWrapper8_copies53 a h).2.2.2.1
  have c1700 := (privateBatchWrapper8_copies53 a h).2.2.2.2.1
  have c1701 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.1
  have c1702 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.1
  have c1703 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.1
  have c1705 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1708 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1709 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1711 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1714 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1715 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1716 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1717 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1718 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1719 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 38) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1692.trans k2
    · exact c1693.trans k2
    · exact c1694.trans k2
    · exact c1695.trans k2
    · exact c1696.trans k2
    · exact c1697.trans k2
    · exact c1698.trans k2
    · exact c1699.trans k2
    · exact c1700.trans k2
    · exact c1701.trans k2
    · exact c1702.trans k2
    · exact c1703.trans k2
    · exact c1704.trans k2
    · exact c1705.trans k2
    · exact c1706.trans k2
    · exact c1707.trans k2
    · exact c1708.trans k2
    · exact c1709.trans k2
    · exact c1710.trans k2
    · exact c1711.trans k2
    · exact c1712.trans k2
    · exact c1713.trans k2
    · exact c1714.trans k2
    · exact c1715.trans k2
    · exact c1716.trans k2
    · exact c1717.trans k2
    · exact c1718.trans k2
    )
  rwa [c1719] at hr

theorem privateBatchWrapper8_f231 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 35) = a (.wire 34 31) + a (.virt 110) := by
  have c1720 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1721 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1722 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_8 := arithEq_of_rows h (row := 34) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_8
  simp only [← c1720, ← c1721, k3, ← c1722] at e_34_8
  have hr := e_34_8
  linear_combination hr

theorem privateBatchWrapper8_f232 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 115)) 32 := by
  have c1723 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1724 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1725 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1726 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1727 := (privateBatchWrapper8_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1728 := (privateBatchWrapper8_copies54 a h).1
  have c1729 := (privateBatchWrapper8_copies54 a h).2.1
  have c1730 := (privateBatchWrapper8_copies54 a h).2.2.1
  have c1731 := (privateBatchWrapper8_copies54 a h).2.2.2.1
  have c1732 := (privateBatchWrapper8_copies54 a h).2.2.2.2.1
  have c1733 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.1
  have c1734 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.1
  have c1735 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.1
  have c1736 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.1
  have c1737 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.1
  have c1738 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1739 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1740 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1741 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1742 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1743 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1744 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1745 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1746 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1747 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1748 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1749 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1750 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 39) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1723.trans k2
    · exact c1724.trans k2
    · exact c1725.trans k2
    · exact c1726.trans k2
    · exact c1727.trans k2
    · exact c1728.trans k2
    · exact c1729.trans k2
    · exact c1730.trans k2
    · exact c1731.trans k2
    · exact c1732.trans k2
    · exact c1733.trans k2
    · exact c1734.trans k2
    · exact c1735.trans k2
    · exact c1736.trans k2
    · exact c1737.trans k2
    · exact c1738.trans k2
    · exact c1739.trans k2
    · exact c1740.trans k2
    · exact c1741.trans k2
    · exact c1742.trans k2
    · exact c1743.trans k2
    · exact c1744.trans k2
    · exact c1745.trans k2
    · exact c1746.trans k2
    · exact c1747.trans k2
    · exact c1748.trans k2
    · exact c1749.trans k2
    )
  rwa [c1750] at hr

theorem privateBatchWrapper8_f233 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 39) = a (.wire 34 35) + a (.virt 115) := by
  have c1751 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1752 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1753 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_9 := arithEq_of_rows h (row := 34) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_9
  simp only [← c1751, ← c1752, k3, ← c1753] at e_34_9
  have hr := e_34_9
  linear_combination hr

theorem privateBatchWrapper8_f234 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 120)) 32 := by
  have c1754 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1755 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1756 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1757 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1758 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1759 := (privateBatchWrapper8_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1760 := (privateBatchWrapper8_copies55 a h).1
  have c1761 := (privateBatchWrapper8_copies55 a h).2.1
  have c1762 := (privateBatchWrapper8_copies55 a h).2.2.1
  have c1763 := (privateBatchWrapper8_copies55 a h).2.2.2.1
  have c1764 := (privateBatchWrapper8_copies55 a h).2.2.2.2.1
  have c1765 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.1
  have c1766 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.1
  have c1767 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.1
  have c1768 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.1
  have c1769 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.1
  have c1770 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1771 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1772 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1773 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1774 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1775 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1776 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1777 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1778 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1779 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1780 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1781 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 40) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1754.trans k2
    · exact c1755.trans k2
    · exact c1756.trans k2
    · exact c1757.trans k2
    · exact c1758.trans k2
    · exact c1759.trans k2
    · exact c1760.trans k2
    · exact c1761.trans k2
    · exact c1762.trans k2
    · exact c1763.trans k2
    · exact c1764.trans k2
    · exact c1765.trans k2
    · exact c1766.trans k2
    · exact c1767.trans k2
    · exact c1768.trans k2
    · exact c1769.trans k2
    · exact c1770.trans k2
    · exact c1771.trans k2
    · exact c1772.trans k2
    · exact c1773.trans k2
    · exact c1774.trans k2
    · exact c1775.trans k2
    · exact c1776.trans k2
    · exact c1777.trans k2
    · exact c1778.trans k2
    · exact c1779.trans k2
    · exact c1780.trans k2
    )
  rwa [c1781] at hr

theorem privateBatchWrapper8_f235 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 43) = a (.wire 34 39) + a (.virt 120) := by
  have c1782 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1783 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1784 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_10 := arithEq_of_rows h (row := 34) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_10
  simp only [← c1782, ← c1783, k3, ← c1784] at e_34_10
  have hr := e_34_10
  linear_combination hr

theorem privateBatchWrapper8_f236 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 125)) 32 := by
  have c1785 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1786 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1787 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1788 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1789 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1790 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1791 := (privateBatchWrapper8_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1792 := (privateBatchWrapper8_copies56 a h).1
  have c1793 := (privateBatchWrapper8_copies56 a h).2.1
  have c1794 := (privateBatchWrapper8_copies56 a h).2.2.1
  have c1795 := (privateBatchWrapper8_copies56 a h).2.2.2.1
  have c1796 := (privateBatchWrapper8_copies56 a h).2.2.2.2.1
  have c1797 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.1
  have c1798 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.1
  have c1799 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.1
  have c1800 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.1
  have c1801 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.1
  have c1802 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1803 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1804 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1805 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1806 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1807 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1808 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1809 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1810 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1811 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1812 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 41) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1785.trans k2
    · exact c1786.trans k2
    · exact c1787.trans k2
    · exact c1788.trans k2
    · exact c1789.trans k2
    · exact c1790.trans k2
    · exact c1791.trans k2
    · exact c1792.trans k2
    · exact c1793.trans k2
    · exact c1794.trans k2
    · exact c1795.trans k2
    · exact c1796.trans k2
    · exact c1797.trans k2
    · exact c1798.trans k2
    · exact c1799.trans k2
    · exact c1800.trans k2
    · exact c1801.trans k2
    · exact c1802.trans k2
    · exact c1803.trans k2
    · exact c1804.trans k2
    · exact c1805.trans k2
    · exact c1806.trans k2
    · exact c1807.trans k2
    · exact c1808.trans k2
    · exact c1809.trans k2
    · exact c1810.trans k2
    · exact c1811.trans k2
    )
  rwa [c1812] at hr

theorem privateBatchWrapper8_f237 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 47) = a (.wire 34 43) + a (.virt 125) := by
  have c1813 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1814 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1815 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_11 := arithEq_of_rows h (row := 34) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_11
  simp only [← c1813, ← c1814, k3, ← c1815] at e_34_11
  have hr := e_34_11
  linear_combination hr

theorem privateBatchWrapper8_f238 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 130)) 32 := by
  have c1816 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1817 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1818 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1819 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1820 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1821 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1822 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1823 := (privateBatchWrapper8_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1824 := (privateBatchWrapper8_copies57 a h).1
  have c1825 := (privateBatchWrapper8_copies57 a h).2.1
  have c1826 := (privateBatchWrapper8_copies57 a h).2.2.1
  have c1827 := (privateBatchWrapper8_copies57 a h).2.2.2.1
  have c1828 := (privateBatchWrapper8_copies57 a h).2.2.2.2.1
  have c1829 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.1
  have c1830 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.1
  have c1831 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.1
  have c1832 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.1
  have c1833 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.1
  have c1834 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1835 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1836 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1837 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1838 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1839 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1840 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1841 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1842 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1843 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 42) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1816.trans k2
    · exact c1817.trans k2
    · exact c1818.trans k2
    · exact c1819.trans k2
    · exact c1820.trans k2
    · exact c1821.trans k2
    · exact c1822.trans k2
    · exact c1823.trans k2
    · exact c1824.trans k2
    · exact c1825.trans k2
    · exact c1826.trans k2
    · exact c1827.trans k2
    · exact c1828.trans k2
    · exact c1829.trans k2
    · exact c1830.trans k2
    · exact c1831.trans k2
    · exact c1832.trans k2
    · exact c1833.trans k2
    · exact c1834.trans k2
    · exact c1835.trans k2
    · exact c1836.trans k2
    · exact c1837.trans k2
    · exact c1838.trans k2
    · exact c1839.trans k2
    · exact c1840.trans k2
    · exact c1841.trans k2
    · exact c1842.trans k2
    )
  rwa [c1843] at hr

theorem privateBatchWrapper8_f239 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 51) = a (.wire 34 47) + a (.virt 130) := by
  have c1844 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1845 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1846 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_12 := arithEq_of_rows h (row := 34) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_12
  simp only [← c1844, ← c1845, k3, ← c1846] at e_34_12
  have hr := e_34_12
  linear_combination hr

theorem privateBatchWrapper8_f240 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 135)) 32 := by
  have c1847 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1848 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1849 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1850 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1851 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1852 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1853 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1854 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1855 := (privateBatchWrapper8_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1856 := (privateBatchWrapper8_copies58 a h).1
  have c1857 := (privateBatchWrapper8_copies58 a h).2.1
  have c1858 := (privateBatchWrapper8_copies58 a h).2.2.1
  have c1859 := (privateBatchWrapper8_copies58 a h).2.2.2.1
  have c1860 := (privateBatchWrapper8_copies58 a h).2.2.2.2.1
  have c1861 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.1
  have c1862 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.1
  have c1863 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.1
  have c1864 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.1
  have c1865 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.1
  have c1866 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1867 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1868 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1869 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1870 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1871 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1872 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1873 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1874 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 43) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1847.trans k2
    · exact c1848.trans k2
    · exact c1849.trans k2
    · exact c1850.trans k2
    · exact c1851.trans k2
    · exact c1852.trans k2
    · exact c1853.trans k2
    · exact c1854.trans k2
    · exact c1855.trans k2
    · exact c1856.trans k2
    · exact c1857.trans k2
    · exact c1858.trans k2
    · exact c1859.trans k2
    · exact c1860.trans k2
    · exact c1861.trans k2
    · exact c1862.trans k2
    · exact c1863.trans k2
    · exact c1864.trans k2
    · exact c1865.trans k2
    · exact c1866.trans k2
    · exact c1867.trans k2
    · exact c1868.trans k2
    · exact c1869.trans k2
    · exact c1870.trans k2
    · exact c1871.trans k2
    · exact c1872.trans k2
    · exact c1873.trans k2
    )
  rwa [c1874] at hr

theorem privateBatchWrapper8_f241 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 55) = a (.wire 34 51) + a (.virt 135) := by
  have c1875 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1876 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1877 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_13 := arithEq_of_rows h (row := 34) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_13
  simp only [← c1875, ← c1876, k3, ← c1877] at e_34_13
  have hr := e_34_13
  linear_combination hr

theorem privateBatchWrapper8_f242 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 140)) 32 := by
  have c1878 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1879 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1880 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1881 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1882 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1883 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1884 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1885 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1886 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1887 := (privateBatchWrapper8_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1888 := (privateBatchWrapper8_copies59 a h).1
  have c1889 := (privateBatchWrapper8_copies59 a h).2.1
  have c1890 := (privateBatchWrapper8_copies59 a h).2.2.1
  have c1891 := (privateBatchWrapper8_copies59 a h).2.2.2.1
  have c1892 := (privateBatchWrapper8_copies59 a h).2.2.2.2.1
  have c1893 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.1
  have c1894 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.1
  have c1895 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.1
  have c1896 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.1
  have c1897 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.1
  have c1898 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1899 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1900 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1901 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1902 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1903 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1904 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1905 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 44) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1878.trans k2
    · exact c1879.trans k2
    · exact c1880.trans k2
    · exact c1881.trans k2
    · exact c1882.trans k2
    · exact c1883.trans k2
    · exact c1884.trans k2
    · exact c1885.trans k2
    · exact c1886.trans k2
    · exact c1887.trans k2
    · exact c1888.trans k2
    · exact c1889.trans k2
    · exact c1890.trans k2
    · exact c1891.trans k2
    · exact c1892.trans k2
    · exact c1893.trans k2
    · exact c1894.trans k2
    · exact c1895.trans k2
    · exact c1896.trans k2
    · exact c1897.trans k2
    · exact c1898.trans k2
    · exact c1899.trans k2
    · exact c1900.trans k2
    · exact c1901.trans k2
    · exact c1902.trans k2
    · exact c1903.trans k2
    · exact c1904.trans k2
    )
  rwa [c1905] at hr

theorem privateBatchWrapper8_f243 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 34 59) = a (.wire 34 55) + a (.virt 140) := by
  have c1906 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1907 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1908 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_14 := arithEq_of_rows h (row := 34) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_14
  simp only [← c1906, ← c1907, k3, ← c1908] at e_34_14
  have hr := e_34_14
  linear_combination hr

theorem privateBatchWrapper8_f244 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 145)) 32 := by
  have c1909 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1910 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1911 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1912 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1913 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1914 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1915 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1916 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1917 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1918 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1919 := (privateBatchWrapper8_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1920 := (privateBatchWrapper8_copies60 a h).1
  have c1921 := (privateBatchWrapper8_copies60 a h).2.1
  have c1922 := (privateBatchWrapper8_copies60 a h).2.2.1
  have c1923 := (privateBatchWrapper8_copies60 a h).2.2.2.1
  have c1924 := (privateBatchWrapper8_copies60 a h).2.2.2.2.1
  have c1925 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.1
  have c1926 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.1
  have c1927 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.1
  have c1928 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.1
  have c1929 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.1
  have c1930 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1931 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1932 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1933 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1934 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1935 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1936 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 45) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1909.trans k2
    · exact c1910.trans k2
    · exact c1911.trans k2
    · exact c1912.trans k2
    · exact c1913.trans k2
    · exact c1914.trans k2
    · exact c1915.trans k2
    · exact c1916.trans k2
    · exact c1917.trans k2
    · exact c1918.trans k2
    · exact c1919.trans k2
    · exact c1920.trans k2
    · exact c1921.trans k2
    · exact c1922.trans k2
    · exact c1923.trans k2
    · exact c1924.trans k2
    · exact c1925.trans k2
    · exact c1926.trans k2
    · exact c1927.trans k2
    · exact c1928.trans k2
    · exact c1929.trans k2
    · exact c1930.trans k2
    · exact c1931.trans k2
    · exact c1932.trans k2
    · exact c1933.trans k2
    · exact c1934.trans k2
    · exact c1935.trans k2
    )
  rwa [c1936] at hr

theorem privateBatchWrapper8_f245 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 3) = a (.wire 34 59) + a (.virt 145) := by
  have c1937 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1938 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1939 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_0 := arithEq_of_rows h (row := 46) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_0
  simp only [← c1937, ← c1938, k3, ← c1939] at e_46_0
  have hr := e_46_0
  linear_combination hr

theorem privateBatchWrapper8_f246 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 150)) 32 := by
  have c1940 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1941 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1942 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1943 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1944 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1945 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1946 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1947 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1948 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1949 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1950 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1951 := (privateBatchWrapper8_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1952 := (privateBatchWrapper8_copies61 a h).1
  have c1953 := (privateBatchWrapper8_copies61 a h).2.1
  have c1954 := (privateBatchWrapper8_copies61 a h).2.2.1
  have c1955 := (privateBatchWrapper8_copies61 a h).2.2.2.1
  have c1956 := (privateBatchWrapper8_copies61 a h).2.2.2.2.1
  have c1957 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.1
  have c1958 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.1
  have c1959 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.1
  have c1960 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.1
  have c1961 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.1
  have c1962 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1963 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1964 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1965 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1966 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1967 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 47) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1940.trans k2
    · exact c1941.trans k2
    · exact c1942.trans k2
    · exact c1943.trans k2
    · exact c1944.trans k2
    · exact c1945.trans k2
    · exact c1946.trans k2
    · exact c1947.trans k2
    · exact c1948.trans k2
    · exact c1949.trans k2
    · exact c1950.trans k2
    · exact c1951.trans k2
    · exact c1952.trans k2
    · exact c1953.trans k2
    · exact c1954.trans k2
    · exact c1955.trans k2
    · exact c1956.trans k2
    · exact c1957.trans k2
    · exact c1958.trans k2
    · exact c1959.trans k2
    · exact c1960.trans k2
    · exact c1961.trans k2
    · exact c1962.trans k2
    · exact c1963.trans k2
    · exact c1964.trans k2
    · exact c1965.trans k2
    · exact c1966.trans k2
    )
  rwa [c1967] at hr

theorem privateBatchWrapper8_f247 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 7) = a (.wire 46 3) + a (.virt 150) := by
  have c1968 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1969 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1970 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_1 := arithEq_of_rows h (row := 46) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_1
  simp only [← c1968, ← c1969, k3, ← c1970] at e_46_1
  have hr := e_46_1
  linear_combination hr

theorem privateBatchWrapper8_f248 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 155)) 32 := by
  have c1971 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1972 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1973 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1974 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1975 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1976 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1977 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1978 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1979 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1980 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1981 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1982 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1983 := (privateBatchWrapper8_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1984 := (privateBatchWrapper8_copies62 a h).1
  have c1985 := (privateBatchWrapper8_copies62 a h).2.1
  have c1986 := (privateBatchWrapper8_copies62 a h).2.2.1
  have c1987 := (privateBatchWrapper8_copies62 a h).2.2.2.1
  have c1988 := (privateBatchWrapper8_copies62 a h).2.2.2.2.1
  have c1989 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.1
  have c1990 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.1
  have c1991 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.1
  have c1992 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.1
  have c1993 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.1
  have c1994 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1995 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1996 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1997 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1998 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 48) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1971.trans k2
    · exact c1972.trans k2
    · exact c1973.trans k2
    · exact c1974.trans k2
    · exact c1975.trans k2
    · exact c1976.trans k2
    · exact c1977.trans k2
    · exact c1978.trans k2
    · exact c1979.trans k2
    · exact c1980.trans k2
    · exact c1981.trans k2
    · exact c1982.trans k2
    · exact c1983.trans k2
    · exact c1984.trans k2
    · exact c1985.trans k2
    · exact c1986.trans k2
    · exact c1987.trans k2
    · exact c1988.trans k2
    · exact c1989.trans k2
    · exact c1990.trans k2
    · exact c1991.trans k2
    · exact c1992.trans k2
    · exact c1993.trans k2
    · exact c1994.trans k2
    · exact c1995.trans k2
    · exact c1996.trans k2
    · exact c1997.trans k2
    )
  rwa [c1998] at hr

theorem privateBatchWrapper8_f249 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 11) = a (.wire 46 7) + a (.virt 155) := by
  have c1999 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2000 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2001 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_2 := arithEq_of_rows h (row := 46) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_2
  simp only [← c1999, ← c2000, k3, ← c2001] at e_46_2
  have hr := e_46_2
  linear_combination hr

theorem privateBatchWrapper8_f250 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 160)) 32 := by
  have c2002 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2003 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2004 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2005 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2006 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2007 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2008 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2009 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2010 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2011 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2012 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2013 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2014 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2015 := (privateBatchWrapper8_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2016 := (privateBatchWrapper8_copies63 a h).1
  have c2017 := (privateBatchWrapper8_copies63 a h).2.1
  have c2018 := (privateBatchWrapper8_copies63 a h).2.2.1
  have c2019 := (privateBatchWrapper8_copies63 a h).2.2.2.1
  have c2020 := (privateBatchWrapper8_copies63 a h).2.2.2.2.1
  have c2021 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.1
  have c2022 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.1
  have c2023 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.1
  have c2024 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.1
  have c2025 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.1
  have c2026 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2027 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2028 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2029 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 49) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2002.trans k2
    · exact c2003.trans k2
    · exact c2004.trans k2
    · exact c2005.trans k2
    · exact c2006.trans k2
    · exact c2007.trans k2
    · exact c2008.trans k2
    · exact c2009.trans k2
    · exact c2010.trans k2
    · exact c2011.trans k2
    · exact c2012.trans k2
    · exact c2013.trans k2
    · exact c2014.trans k2
    · exact c2015.trans k2
    · exact c2016.trans k2
    · exact c2017.trans k2
    · exact c2018.trans k2
    · exact c2019.trans k2
    · exact c2020.trans k2
    · exact c2021.trans k2
    · exact c2022.trans k2
    · exact c2023.trans k2
    · exact c2024.trans k2
    · exact c2025.trans k2
    · exact c2026.trans k2
    · exact c2027.trans k2
    · exact c2028.trans k2
    )
  rwa [c2029] at hr

theorem privateBatchWrapper8_f251 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 15) = a (.wire 46 11) + a (.virt 160) := by
  have c2030 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2031 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2032 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_3 := arithEq_of_rows h (row := 46) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_3
  simp only [← c2030, ← c2031, k3, ← c2032] at e_46_3
  have hr := e_46_3
  linear_combination hr

theorem privateBatchWrapper8_f252 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 165)) 32 := by
  have c2033 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2034 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2035 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2036 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2037 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2038 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2039 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2040 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2046 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2047 := (privateBatchWrapper8_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2048 := (privateBatchWrapper8_copies64 a h).1
  have c2049 := (privateBatchWrapper8_copies64 a h).2.1
  have c2050 := (privateBatchWrapper8_copies64 a h).2.2.1
  have c2051 := (privateBatchWrapper8_copies64 a h).2.2.2.1
  have c2052 := (privateBatchWrapper8_copies64 a h).2.2.2.2.1
  have c2053 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.1
  have c2054 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.1
  have c2055 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.1
  have c2056 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.1
  have c2057 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.1
  have c2058 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2059 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2060 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 50) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2033.trans k2
    · exact c2034.trans k2
    · exact c2035.trans k2
    · exact c2036.trans k2
    · exact c2037.trans k2
    · exact c2038.trans k2
    · exact c2039.trans k2
    · exact c2040.trans k2
    · exact c2041.trans k2
    · exact c2042.trans k2
    · exact c2043.trans k2
    · exact c2044.trans k2
    · exact c2045.trans k2
    · exact c2046.trans k2
    · exact c2047.trans k2
    · exact c2048.trans k2
    · exact c2049.trans k2
    · exact c2050.trans k2
    · exact c2051.trans k2
    · exact c2052.trans k2
    · exact c2053.trans k2
    · exact c2054.trans k2
    · exact c2055.trans k2
    · exact c2056.trans k2
    · exact c2057.trans k2
    · exact c2058.trans k2
    · exact c2059.trans k2
    )
  rwa [c2060] at hr

theorem privateBatchWrapper8_f253 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 19) = a (.wire 46 15) + a (.virt 165) := by
  have c2061 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2062 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2063 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_4 := arithEq_of_rows h (row := 46) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_4
  simp only [← c2061, ← c2062, k3, ← c2063] at e_46_4
  have hr := e_46_4
  linear_combination hr

theorem privateBatchWrapper8_f254 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 170)) 32 := by
  have c2064 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2065 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2066 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2067 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2068 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2069 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2070 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2071 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2072 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2073 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2074 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2075 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2076 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2077 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2078 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2079 := (privateBatchWrapper8_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2080 := (privateBatchWrapper8_copies65 a h).1
  have c2081 := (privateBatchWrapper8_copies65 a h).2.1
  have c2082 := (privateBatchWrapper8_copies65 a h).2.2.1
  have c2083 := (privateBatchWrapper8_copies65 a h).2.2.2.1
  have c2084 := (privateBatchWrapper8_copies65 a h).2.2.2.2.1
  have c2085 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.1
  have c2086 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.1
  have c2087 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.1
  have c2088 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.1
  have c2089 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.1
  have c2090 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2091 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 51) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2064.trans k2
    · exact c2065.trans k2
    · exact c2066.trans k2
    · exact c2067.trans k2
    · exact c2068.trans k2
    · exact c2069.trans k2
    · exact c2070.trans k2
    · exact c2071.trans k2
    · exact c2072.trans k2
    · exact c2073.trans k2
    · exact c2074.trans k2
    · exact c2075.trans k2
    · exact c2076.trans k2
    · exact c2077.trans k2
    · exact c2078.trans k2
    · exact c2079.trans k2
    · exact c2080.trans k2
    · exact c2081.trans k2
    · exact c2082.trans k2
    · exact c2083.trans k2
    · exact c2084.trans k2
    · exact c2085.trans k2
    · exact c2086.trans k2
    · exact c2087.trans k2
    · exact c2088.trans k2
    · exact c2089.trans k2
    · exact c2090.trans k2
    )
  rwa [c2091] at hr

theorem privateBatchWrapper8_f255 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 23) = a (.wire 46 19) + a (.virt 170) := by
  have c2092 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2093 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2094 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_5 := arithEq_of_rows h (row := 46) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_5
  simp only [← c2092, ← c2093, k3, ← c2094] at e_46_5
  have hr := e_46_5
  linear_combination hr

theorem privateBatchWrapper8_f256 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 175)) 32 := by
  have c2095 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2096 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2097 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2098 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2099 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2100 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2101 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2102 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2103 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2104 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2105 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2106 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2107 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2108 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2109 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2110 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2111 := (privateBatchWrapper8_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2112 := (privateBatchWrapper8_copies66 a h).1
  have c2113 := (privateBatchWrapper8_copies66 a h).2.1
  have c2114 := (privateBatchWrapper8_copies66 a h).2.2.1
  have c2115 := (privateBatchWrapper8_copies66 a h).2.2.2.1
  have c2116 := (privateBatchWrapper8_copies66 a h).2.2.2.2.1
  have c2117 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.1
  have c2118 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.1
  have c2119 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.1
  have c2120 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.1
  have c2121 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.1
  have c2122 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 52) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2095.trans k2
    · exact c2096.trans k2
    · exact c2097.trans k2
    · exact c2098.trans k2
    · exact c2099.trans k2
    · exact c2100.trans k2
    · exact c2101.trans k2
    · exact c2102.trans k2
    · exact c2103.trans k2
    · exact c2104.trans k2
    · exact c2105.trans k2
    · exact c2106.trans k2
    · exact c2107.trans k2
    · exact c2108.trans k2
    · exact c2109.trans k2
    · exact c2110.trans k2
    · exact c2111.trans k2
    · exact c2112.trans k2
    · exact c2113.trans k2
    · exact c2114.trans k2
    · exact c2115.trans k2
    · exact c2116.trans k2
    · exact c2117.trans k2
    · exact c2118.trans k2
    · exact c2119.trans k2
    · exact c2120.trans k2
    · exact c2121.trans k2
    )
  rwa [c2122] at hr

theorem privateBatchWrapper8_f257 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 27) = a (.wire 46 23) + a (.virt 175) := by
  have c2123 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2124 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2125 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_6 := arithEq_of_rows h (row := 46) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c2123, ← c2124, k3, ← c2125] at e_46_6
  have hr := e_46_6
  linear_combination hr

theorem privateBatchWrapper8_f258 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 180)) 32 := by
  have c2126 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2127 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2128 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2129 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2130 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2131 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2132 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2133 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2134 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2135 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2136 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2137 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2138 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2139 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2140 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2141 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2142 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2143 := (privateBatchWrapper8_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2144 := (privateBatchWrapper8_copies67 a h).1
  have c2145 := (privateBatchWrapper8_copies67 a h).2.1
  have c2146 := (privateBatchWrapper8_copies67 a h).2.2.1
  have c2147 := (privateBatchWrapper8_copies67 a h).2.2.2.1
  have c2148 := (privateBatchWrapper8_copies67 a h).2.2.2.2.1
  have c2149 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.1
  have c2150 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.1
  have c2151 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.1
  have c2152 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.1
  have c2153 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 53) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2126.trans k2
    · exact c2127.trans k2
    · exact c2128.trans k2
    · exact c2129.trans k2
    · exact c2130.trans k2
    · exact c2131.trans k2
    · exact c2132.trans k2
    · exact c2133.trans k2
    · exact c2134.trans k2
    · exact c2135.trans k2
    · exact c2136.trans k2
    · exact c2137.trans k2
    · exact c2138.trans k2
    · exact c2139.trans k2
    · exact c2140.trans k2
    · exact c2141.trans k2
    · exact c2142.trans k2
    · exact c2143.trans k2
    · exact c2144.trans k2
    · exact c2145.trans k2
    · exact c2146.trans k2
    · exact c2147.trans k2
    · exact c2148.trans k2
    · exact c2149.trans k2
    · exact c2150.trans k2
    · exact c2151.trans k2
    · exact c2152.trans k2
    )
  rwa [c2153] at hr

theorem privateBatchWrapper8_f259 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 31) = a (.wire 46 27) + a (.virt 180) := by
  have c2154 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2155 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2156 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_7 := arithEq_of_rows h (row := 46) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_7
  simp only [← c2154, ← c2155, k3, ← c2156] at e_46_7
  have hr := e_46_7
  linear_combination hr

theorem privateBatchWrapper8_f260 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 185)) 32 := by
  have c2157 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2158 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2159 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2160 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2161 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2162 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2163 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2164 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2165 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2166 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2167 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2168 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2169 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2170 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2171 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2172 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2173 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2174 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2175 := (privateBatchWrapper8_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2176 := (privateBatchWrapper8_copies68 a h).1
  have c2177 := (privateBatchWrapper8_copies68 a h).2.1
  have c2178 := (privateBatchWrapper8_copies68 a h).2.2.1
  have c2179 := (privateBatchWrapper8_copies68 a h).2.2.2.1
  have c2180 := (privateBatchWrapper8_copies68 a h).2.2.2.2.1
  have c2181 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.1
  have c2182 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.1
  have c2183 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.1
  have c2184 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 54) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2157.trans k2
    · exact c2158.trans k2
    · exact c2159.trans k2
    · exact c2160.trans k2
    · exact c2161.trans k2
    · exact c2162.trans k2
    · exact c2163.trans k2
    · exact c2164.trans k2
    · exact c2165.trans k2
    · exact c2166.trans k2
    · exact c2167.trans k2
    · exact c2168.trans k2
    · exact c2169.trans k2
    · exact c2170.trans k2
    · exact c2171.trans k2
    · exact c2172.trans k2
    · exact c2173.trans k2
    · exact c2174.trans k2
    · exact c2175.trans k2
    · exact c2176.trans k2
    · exact c2177.trans k2
    · exact c2178.trans k2
    · exact c2179.trans k2
    · exact c2180.trans k2
    · exact c2181.trans k2
    · exact c2182.trans k2
    · exact c2183.trans k2
    )
  rwa [c2184] at hr

theorem privateBatchWrapper8_f261 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 35) = a (.wire 46 31) + a (.virt 185) := by
  have c2185 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.1
  have c2186 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2187 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_8 := arithEq_of_rows h (row := 46) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_8
  simp only [← c2185, ← c2186, k3, ← c2187] at e_46_8
  have hr := e_46_8
  linear_combination hr

theorem privateBatchWrapper8_f262 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 190)) 32 := by
  have c2188 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2189 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2190 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2191 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2192 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2193 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2194 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2195 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2196 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2197 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2198 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2199 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2200 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2201 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2202 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2203 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2204 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2205 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2206 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2207 := (privateBatchWrapper8_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2208 := (privateBatchWrapper8_copies69 a h).1
  have c2209 := (privateBatchWrapper8_copies69 a h).2.1
  have c2210 := (privateBatchWrapper8_copies69 a h).2.2.1
  have c2211 := (privateBatchWrapper8_copies69 a h).2.2.2.1
  have c2212 := (privateBatchWrapper8_copies69 a h).2.2.2.2.1
  have c2213 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.1
  have c2214 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.1
  have c2215 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 55) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2188.trans k2
    · exact c2189.trans k2
    · exact c2190.trans k2
    · exact c2191.trans k2
    · exact c2192.trans k2
    · exact c2193.trans k2
    · exact c2194.trans k2
    · exact c2195.trans k2
    · exact c2196.trans k2
    · exact c2197.trans k2
    · exact c2198.trans k2
    · exact c2199.trans k2
    · exact c2200.trans k2
    · exact c2201.trans k2
    · exact c2202.trans k2
    · exact c2203.trans k2
    · exact c2204.trans k2
    · exact c2205.trans k2
    · exact c2206.trans k2
    · exact c2207.trans k2
    · exact c2208.trans k2
    · exact c2209.trans k2
    · exact c2210.trans k2
    · exact c2211.trans k2
    · exact c2212.trans k2
    · exact c2213.trans k2
    · exact c2214.trans k2
    )
  rwa [c2215] at hr

theorem privateBatchWrapper8_f263 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 39) = a (.wire 46 35) + a (.virt 190) := by
  have c2216 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.1
  have c2217 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.1
  have c2218 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_9 := arithEq_of_rows h (row := 46) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_9
  simp only [← c2216, ← c2217, k3, ← c2218] at e_46_9
  have hr := e_46_9
  linear_combination hr

theorem privateBatchWrapper8_f264 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 195)) 32 := by
  have c2219 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2220 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2221 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2222 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2223 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2224 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2225 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2226 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2227 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2228 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2229 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2230 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2231 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2232 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2233 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2234 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2235 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2236 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2237 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2238 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2239 := (privateBatchWrapper8_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2240 := (privateBatchWrapper8_copies70 a h).1
  have c2241 := (privateBatchWrapper8_copies70 a h).2.1
  have c2242 := (privateBatchWrapper8_copies70 a h).2.2.1
  have c2243 := (privateBatchWrapper8_copies70 a h).2.2.2.1
  have c2244 := (privateBatchWrapper8_copies70 a h).2.2.2.2.1
  have c2245 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.1
  have c2246 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 56) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2219.trans k2
    · exact c2220.trans k2
    · exact c2221.trans k2
    · exact c2222.trans k2
    · exact c2223.trans k2
    · exact c2224.trans k2
    · exact c2225.trans k2
    · exact c2226.trans k2
    · exact c2227.trans k2
    · exact c2228.trans k2
    · exact c2229.trans k2
    · exact c2230.trans k2
    · exact c2231.trans k2
    · exact c2232.trans k2
    · exact c2233.trans k2
    · exact c2234.trans k2
    · exact c2235.trans k2
    · exact c2236.trans k2
    · exact c2237.trans k2
    · exact c2238.trans k2
    · exact c2239.trans k2
    · exact c2240.trans k2
    · exact c2241.trans k2
    · exact c2242.trans k2
    · exact c2243.trans k2
    · exact c2244.trans k2
    · exact c2245.trans k2
    )
  rwa [c2246] at hr

theorem privateBatchWrapper8_f265 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 43) = a (.wire 46 39) + a (.virt 195) := by
  have c2247 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.1
  have c2248 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.1
  have c2249 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_10 := arithEq_of_rows h (row := 46) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_10
  simp only [← c2247, ← c2248, k3, ← c2249] at e_46_10
  have hr := e_46_10
  linear_combination hr

theorem privateBatchWrapper8_f266 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 200)) 32 := by
  have c2250 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2251 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2252 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2253 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2254 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2255 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2256 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2257 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2258 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2259 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2260 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2261 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2262 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2263 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2264 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2265 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2266 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2267 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2268 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2269 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2270 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2271 := (privateBatchWrapper8_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2272 := (privateBatchWrapper8_copies71 a h).1
  have c2273 := (privateBatchWrapper8_copies71 a h).2.1
  have c2274 := (privateBatchWrapper8_copies71 a h).2.2.1
  have c2275 := (privateBatchWrapper8_copies71 a h).2.2.2.1
  have c2276 := (privateBatchWrapper8_copies71 a h).2.2.2.2.1
  have c2277 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 57) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2250.trans k2
    · exact c2251.trans k2
    · exact c2252.trans k2
    · exact c2253.trans k2
    · exact c2254.trans k2
    · exact c2255.trans k2
    · exact c2256.trans k2
    · exact c2257.trans k2
    · exact c2258.trans k2
    · exact c2259.trans k2
    · exact c2260.trans k2
    · exact c2261.trans k2
    · exact c2262.trans k2
    · exact c2263.trans k2
    · exact c2264.trans k2
    · exact c2265.trans k2
    · exact c2266.trans k2
    · exact c2267.trans k2
    · exact c2268.trans k2
    · exact c2269.trans k2
    · exact c2270.trans k2
    · exact c2271.trans k2
    · exact c2272.trans k2
    · exact c2273.trans k2
    · exact c2274.trans k2
    · exact c2275.trans k2
    · exact c2276.trans k2
    )
  rwa [c2277] at hr

theorem privateBatchWrapper8_f267 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 47) = a (.wire 46 43) + a (.virt 200) := by
  have c2278 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.1
  have c2279 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.1
  have c2280 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_11 := arithEq_of_rows h (row := 46) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_11
  simp only [← c2278, ← c2279, k3, ← c2280] at e_46_11
  have hr := e_46_11
  linear_combination hr

theorem privateBatchWrapper8_f268 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 205)) 32 := by
  have c2281 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.1
  have c2282 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2283 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2284 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2285 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2286 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2287 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2288 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2289 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2290 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2291 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2292 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2293 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2294 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2295 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2296 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2297 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2298 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2299 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2300 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2301 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2302 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2303 := (privateBatchWrapper8_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2304 := (privateBatchWrapper8_copies72 a h).1
  have c2305 := (privateBatchWrapper8_copies72 a h).2.1
  have c2306 := (privateBatchWrapper8_copies72 a h).2.2.1
  have c2307 := (privateBatchWrapper8_copies72 a h).2.2.2.1
  have c2308 := (privateBatchWrapper8_copies72 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 58) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2281.trans k2
    · exact c2282.trans k2
    · exact c2283.trans k2
    · exact c2284.trans k2
    · exact c2285.trans k2
    · exact c2286.trans k2
    · exact c2287.trans k2
    · exact c2288.trans k2
    · exact c2289.trans k2
    · exact c2290.trans k2
    · exact c2291.trans k2
    · exact c2292.trans k2
    · exact c2293.trans k2
    · exact c2294.trans k2
    · exact c2295.trans k2
    · exact c2296.trans k2
    · exact c2297.trans k2
    · exact c2298.trans k2
    · exact c2299.trans k2
    · exact c2300.trans k2
    · exact c2301.trans k2
    · exact c2302.trans k2
    · exact c2303.trans k2
    · exact c2304.trans k2
    · exact c2305.trans k2
    · exact c2306.trans k2
    · exact c2307.trans k2
    )
  rwa [c2308] at hr

theorem privateBatchWrapper8_f269 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 51) = a (.wire 46 47) + a (.virt 205) := by
  have c2309 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.1
  have c2310 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.1
  have c2311 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_12 := arithEq_of_rows h (row := 46) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_12
  simp only [← c2309, ← c2310, k3, ← c2311] at e_46_12
  have hr := e_46_12
  linear_combination hr

theorem privateBatchWrapper8_f270 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 210)) 32 := by
  have c2312 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.1
  have c2313 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.1
  have c2314 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2315 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2316 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2317 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2318 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2319 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2320 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2321 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2322 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2323 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2324 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2325 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2326 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2327 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2328 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2329 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2330 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2331 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2332 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2333 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2334 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2335 := (privateBatchWrapper8_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2336 := (privateBatchWrapper8_copies73 a h).1
  have c2337 := (privateBatchWrapper8_copies73 a h).2.1
  have c2338 := (privateBatchWrapper8_copies73 a h).2.2.1
  have c2339 := (privateBatchWrapper8_copies73 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 59) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2312.trans k2
    · exact c2313.trans k2
    · exact c2314.trans k2
    · exact c2315.trans k2
    · exact c2316.trans k2
    · exact c2317.trans k2
    · exact c2318.trans k2
    · exact c2319.trans k2
    · exact c2320.trans k2
    · exact c2321.trans k2
    · exact c2322.trans k2
    · exact c2323.trans k2
    · exact c2324.trans k2
    · exact c2325.trans k2
    · exact c2326.trans k2
    · exact c2327.trans k2
    · exact c2328.trans k2
    · exact c2329.trans k2
    · exact c2330.trans k2
    · exact c2331.trans k2
    · exact c2332.trans k2
    · exact c2333.trans k2
    · exact c2334.trans k2
    · exact c2335.trans k2
    · exact c2336.trans k2
    · exact c2337.trans k2
    · exact c2338.trans k2
    )
  rwa [c2339] at hr

theorem privateBatchWrapper8_f271 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 55) = a (.wire 46 51) + a (.virt 210) := by
  have c2340 := (privateBatchWrapper8_copies73 a h).2.2.2.2.1
  have c2341 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.1
  have c2342 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_13 := arithEq_of_rows h (row := 46) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_13
  simp only [← c2340, ← c2341, k3, ← c2342] at e_46_13
  have hr := e_46_13
  linear_combination hr

theorem privateBatchWrapper8_f272 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 215)) 32 := by
  have c2343 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.1
  have c2344 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.1
  have c2345 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.1
  have c2346 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2347 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2348 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2349 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2350 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2351 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2352 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2353 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2354 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2355 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2356 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2357 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2358 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2359 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2360 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2361 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2362 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2363 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2364 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2365 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2366 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2367 := (privateBatchWrapper8_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2368 := (privateBatchWrapper8_copies74 a h).1
  have c2369 := (privateBatchWrapper8_copies74 a h).2.1
  have c2370 := (privateBatchWrapper8_copies74 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 60) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2343.trans k2
    · exact c2344.trans k2
    · exact c2345.trans k2
    · exact c2346.trans k2
    · exact c2347.trans k2
    · exact c2348.trans k2
    · exact c2349.trans k2
    · exact c2350.trans k2
    · exact c2351.trans k2
    · exact c2352.trans k2
    · exact c2353.trans k2
    · exact c2354.trans k2
    · exact c2355.trans k2
    · exact c2356.trans k2
    · exact c2357.trans k2
    · exact c2358.trans k2
    · exact c2359.trans k2
    · exact c2360.trans k2
    · exact c2361.trans k2
    · exact c2362.trans k2
    · exact c2363.trans k2
    · exact c2364.trans k2
    · exact c2365.trans k2
    · exact c2366.trans k2
    · exact c2367.trans k2
    · exact c2368.trans k2
    · exact c2369.trans k2
    )
  rwa [c2370] at hr

theorem privateBatchWrapper8_f273 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 46 59) = a (.wire 46 55) + a (.virt 215) := by
  have c2371 := (privateBatchWrapper8_copies74 a h).2.2.2.1
  have c2372 := (privateBatchWrapper8_copies74 a h).2.2.2.2.1
  have c2373 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_46_14 := arithEq_of_rows h (row := 46) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_14
  simp only [← c2371, ← c2372, k3, ← c2373] at e_46_14
  have hr := e_46_14
  linear_combination hr

theorem privateBatchWrapper8_f274 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 220)) 32 := by
  have c2374 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.1
  have c2375 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.1
  have c2376 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.1
  have c2377 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.1
  have c2378 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2379 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2380 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2381 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2382 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2383 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2384 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2385 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2386 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2387 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2388 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2389 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2390 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2391 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2392 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2393 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2394 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2395 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2396 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2397 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2398 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2399 := (privateBatchWrapper8_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2400 := (privateBatchWrapper8_copies75 a h).1
  have c2401 := (privateBatchWrapper8_copies75 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 61) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2374.trans k2
    · exact c2375.trans k2
    · exact c2376.trans k2
    · exact c2377.trans k2
    · exact c2378.trans k2
    · exact c2379.trans k2
    · exact c2380.trans k2
    · exact c2381.trans k2
    · exact c2382.trans k2
    · exact c2383.trans k2
    · exact c2384.trans k2
    · exact c2385.trans k2
    · exact c2386.trans k2
    · exact c2387.trans k2
    · exact c2388.trans k2
    · exact c2389.trans k2
    · exact c2390.trans k2
    · exact c2391.trans k2
    · exact c2392.trans k2
    · exact c2393.trans k2
    · exact c2394.trans k2
    · exact c2395.trans k2
    · exact c2396.trans k2
    · exact c2397.trans k2
    · exact c2398.trans k2
    · exact c2399.trans k2
    · exact c2400.trans k2
    )
  rwa [c2401] at hr

theorem privateBatchWrapper8_f275 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 3) = a (.wire 46 59) + a (.virt 220) := by
  have c2402 := (privateBatchWrapper8_copies75 a h).2.2.1
  have c2403 := (privateBatchWrapper8_copies75 a h).2.2.2.1
  have c2404 := (privateBatchWrapper8_copies75 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_0 := arithEq_of_rows h (row := 62) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_0
  simp only [← c2402, ← c2403, k3, ← c2404] at e_62_0
  have hr := e_62_0
  linear_combination hr

theorem privateBatchWrapper8_f276 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 225)) 32 := by
  have c2405 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.1
  have c2406 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.1
  have c2407 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.1
  have c2408 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.1
  have c2409 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.1
  have c2410 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2411 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2412 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2413 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2414 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2415 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2416 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2417 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2418 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2419 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2420 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2421 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2422 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2423 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2424 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2425 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2426 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2427 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2428 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2429 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2430 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2431 := (privateBatchWrapper8_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2432 := (privateBatchWrapper8_copies76 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 63) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2405.trans k2
    · exact c2406.trans k2
    · exact c2407.trans k2
    · exact c2408.trans k2
    · exact c2409.trans k2
    · exact c2410.trans k2
    · exact c2411.trans k2
    · exact c2412.trans k2
    · exact c2413.trans k2
    · exact c2414.trans k2
    · exact c2415.trans k2
    · exact c2416.trans k2
    · exact c2417.trans k2
    · exact c2418.trans k2
    · exact c2419.trans k2
    · exact c2420.trans k2
    · exact c2421.trans k2
    · exact c2422.trans k2
    · exact c2423.trans k2
    · exact c2424.trans k2
    · exact c2425.trans k2
    · exact c2426.trans k2
    · exact c2427.trans k2
    · exact c2428.trans k2
    · exact c2429.trans k2
    · exact c2430.trans k2
    · exact c2431.trans k2
    )
  rwa [c2432] at hr

theorem privateBatchWrapper8_f277 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 7) = a (.wire 62 3) + a (.virt 225) := by
  have c2433 := (privateBatchWrapper8_copies76 a h).2.1
  have c2434 := (privateBatchWrapper8_copies76 a h).2.2.1
  have c2435 := (privateBatchWrapper8_copies76 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_1 := arithEq_of_rows h (row := 62) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_1
  simp only [← c2433, ← c2434, k3, ← c2435] at e_62_1
  have hr := e_62_1
  linear_combination hr

theorem privateBatchWrapper8_f278 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 230)) 32 := by
  have c2436 := (privateBatchWrapper8_copies76 a h).2.2.2.2.1
  have c2437 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.1
  have c2438 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.1
  have c2439 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.1
  have c2440 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.1
  have c2441 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.1
  have c2442 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2443 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2444 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2445 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2446 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2447 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2448 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2449 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2450 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2451 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2452 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2453 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2454 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2455 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2456 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2457 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2458 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2459 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2460 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2461 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2462 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2463 := (privateBatchWrapper8_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 64) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2436.trans k2
    · exact c2437.trans k2
    · exact c2438.trans k2
    · exact c2439.trans k2
    · exact c2440.trans k2
    · exact c2441.trans k2
    · exact c2442.trans k2
    · exact c2443.trans k2
    · exact c2444.trans k2
    · exact c2445.trans k2
    · exact c2446.trans k2
    · exact c2447.trans k2
    · exact c2448.trans k2
    · exact c2449.trans k2
    · exact c2450.trans k2
    · exact c2451.trans k2
    · exact c2452.trans k2
    · exact c2453.trans k2
    · exact c2454.trans k2
    · exact c2455.trans k2
    · exact c2456.trans k2
    · exact c2457.trans k2
    · exact c2458.trans k2
    · exact c2459.trans k2
    · exact c2460.trans k2
    · exact c2461.trans k2
    · exact c2462.trans k2
    )
  rwa [c2463] at hr

theorem privateBatchWrapper8_f279 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 11) = a (.wire 62 7) + a (.virt 230) := by
  have c2464 := (privateBatchWrapper8_copies77 a h).1
  have c2465 := (privateBatchWrapper8_copies77 a h).2.1
  have c2466 := (privateBatchWrapper8_copies77 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_2 := arithEq_of_rows h (row := 62) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_2
  simp only [← c2464, ← c2465, k3, ← c2466] at e_62_2
  have hr := e_62_2
  linear_combination hr

theorem privateBatchWrapper8_f280 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 235)) 32 := by
  have c2467 := (privateBatchWrapper8_copies77 a h).2.2.2.1
  have c2468 := (privateBatchWrapper8_copies77 a h).2.2.2.2.1
  have c2469 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.1
  have c2470 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.1
  have c2471 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.1
  have c2472 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.1
  have c2473 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.1
  have c2474 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2475 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2476 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2477 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2478 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2479 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2480 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2481 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2482 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2483 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2484 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2485 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2486 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2487 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2488 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2489 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2490 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2491 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2492 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2493 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2494 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 65) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2467.trans k2
    · exact c2468.trans k2
    · exact c2469.trans k2
    · exact c2470.trans k2
    · exact c2471.trans k2
    · exact c2472.trans k2
    · exact c2473.trans k2
    · exact c2474.trans k2
    · exact c2475.trans k2
    · exact c2476.trans k2
    · exact c2477.trans k2
    · exact c2478.trans k2
    · exact c2479.trans k2
    · exact c2480.trans k2
    · exact c2481.trans k2
    · exact c2482.trans k2
    · exact c2483.trans k2
    · exact c2484.trans k2
    · exact c2485.trans k2
    · exact c2486.trans k2
    · exact c2487.trans k2
    · exact c2488.trans k2
    · exact c2489.trans k2
    · exact c2490.trans k2
    · exact c2491.trans k2
    · exact c2492.trans k2
    · exact c2493.trans k2
    )
  rwa [c2494] at hr

theorem privateBatchWrapper8_f281 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 15) = a (.wire 62 11) + a (.virt 235) := by
  have c2495 := (privateBatchWrapper8_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2496 := (privateBatchWrapper8_copies78 a h).1
  have c2497 := (privateBatchWrapper8_copies78 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_3 := arithEq_of_rows h (row := 62) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_3
  simp only [← c2495, ← c2496, k3, ← c2497] at e_62_3
  have hr := e_62_3
  linear_combination hr

theorem privateBatchWrapper8_f282 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 240)) 32 := by
  have c2498 := (privateBatchWrapper8_copies78 a h).2.2.1
  have c2499 := (privateBatchWrapper8_copies78 a h).2.2.2.1
  have c2500 := (privateBatchWrapper8_copies78 a h).2.2.2.2.1
  have c2501 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.1
  have c2502 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.1
  have c2503 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.1
  have c2504 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.1
  have c2505 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.1
  have c2506 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2507 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2508 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2509 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2510 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2511 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2512 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2513 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2514 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2515 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2516 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2517 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2518 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2519 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2522 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2524 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2525 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 66) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2498.trans k2
    · exact c2499.trans k2
    · exact c2500.trans k2
    · exact c2501.trans k2
    · exact c2502.trans k2
    · exact c2503.trans k2
    · exact c2504.trans k2
    · exact c2505.trans k2
    · exact c2506.trans k2
    · exact c2507.trans k2
    · exact c2508.trans k2
    · exact c2509.trans k2
    · exact c2510.trans k2
    · exact c2511.trans k2
    · exact c2512.trans k2
    · exact c2513.trans k2
    · exact c2514.trans k2
    · exact c2515.trans k2
    · exact c2516.trans k2
    · exact c2517.trans k2
    · exact c2518.trans k2
    · exact c2519.trans k2
    · exact c2520.trans k2
    · exact c2521.trans k2
    · exact c2522.trans k2
    · exact c2523.trans k2
    · exact c2524.trans k2
    )
  rwa [c2525] at hr

theorem privateBatchWrapper8_f283 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 19) = a (.wire 62 15) + a (.virt 240) := by
  have c2526 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2527 := (privateBatchWrapper8_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2528 := (privateBatchWrapper8_copies79 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_4 := arithEq_of_rows h (row := 62) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_4
  simp only [← c2526, ← c2527, k3, ← c2528] at e_62_4
  have hr := e_62_4
  linear_combination hr

theorem privateBatchWrapper8_f284 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 245)) 32 := by
  have c2529 := (privateBatchWrapper8_copies79 a h).2.1
  have c2530 := (privateBatchWrapper8_copies79 a h).2.2.1
  have c2531 := (privateBatchWrapper8_copies79 a h).2.2.2.1
  have c2532 := (privateBatchWrapper8_copies79 a h).2.2.2.2.1
  have c2533 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.1
  have c2534 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.1
  have c2535 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.1
  have c2536 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.1
  have c2537 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.1
  have c2538 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2539 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2540 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2541 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2542 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2543 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2544 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2545 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2546 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2547 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2548 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2549 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2550 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2551 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2552 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2553 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2554 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2555 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2556 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 67) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2529.trans k2
    · exact c2530.trans k2
    · exact c2531.trans k2
    · exact c2532.trans k2
    · exact c2533.trans k2
    · exact c2534.trans k2
    · exact c2535.trans k2
    · exact c2536.trans k2
    · exact c2537.trans k2
    · exact c2538.trans k2
    · exact c2539.trans k2
    · exact c2540.trans k2
    · exact c2541.trans k2
    · exact c2542.trans k2
    · exact c2543.trans k2
    · exact c2544.trans k2
    · exact c2545.trans k2
    · exact c2546.trans k2
    · exact c2547.trans k2
    · exact c2548.trans k2
    · exact c2549.trans k2
    · exact c2550.trans k2
    · exact c2551.trans k2
    · exact c2552.trans k2
    · exact c2553.trans k2
    · exact c2554.trans k2
    · exact c2555.trans k2
    )
  rwa [c2556] at hr

theorem privateBatchWrapper8_f285 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 23) = a (.wire 62 19) + a (.virt 245) := by
  have c2557 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2558 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2559 := (privateBatchWrapper8_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_5 := arithEq_of_rows h (row := 62) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_5
  simp only [← c2557, ← c2558, k3, ← c2559] at e_62_5
  have hr := e_62_5
  linear_combination hr

theorem privateBatchWrapper8_f286 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 250)) 32 := by
  have c2560 := (privateBatchWrapper8_copies80 a h).1
  have c2561 := (privateBatchWrapper8_copies80 a h).2.1
  have c2562 := (privateBatchWrapper8_copies80 a h).2.2.1
  have c2563 := (privateBatchWrapper8_copies80 a h).2.2.2.1
  have c2564 := (privateBatchWrapper8_copies80 a h).2.2.2.2.1
  have c2565 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.1
  have c2566 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.1
  have c2567 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.1
  have c2568 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.1
  have c2569 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.1
  have c2570 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2571 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2572 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2573 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2574 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2575 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2576 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2577 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2578 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2579 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2580 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2581 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2582 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2583 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2584 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2585 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2586 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2587 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 68) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2560.trans k2
    · exact c2561.trans k2
    · exact c2562.trans k2
    · exact c2563.trans k2
    · exact c2564.trans k2
    · exact c2565.trans k2
    · exact c2566.trans k2
    · exact c2567.trans k2
    · exact c2568.trans k2
    · exact c2569.trans k2
    · exact c2570.trans k2
    · exact c2571.trans k2
    · exact c2572.trans k2
    · exact c2573.trans k2
    · exact c2574.trans k2
    · exact c2575.trans k2
    · exact c2576.trans k2
    · exact c2577.trans k2
    · exact c2578.trans k2
    · exact c2579.trans k2
    · exact c2580.trans k2
    · exact c2581.trans k2
    · exact c2582.trans k2
    · exact c2583.trans k2
    · exact c2584.trans k2
    · exact c2585.trans k2
    · exact c2586.trans k2
    )
  rwa [c2587] at hr

theorem privateBatchWrapper8_f287 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 62 27) = a (.wire 62 23) + a (.virt 250) := by
  have c2588 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2589 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2590 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_62_6 := arithEq_of_rows h (row := 62) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_6
  simp only [← c2588, ← c2589, k3, ← c2590] at e_62_6
  have hr := e_62_6
  linear_combination hr

theorem privateBatchWrapper8_f288 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.virt 90)) 14 := by
  have c2591 := (privateBatchWrapper8_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2592 := (privateBatchWrapper8_copies81 a h).1
  have c2593 := (privateBatchWrapper8_copies81 a h).2.1
  have c2594 := (privateBatchWrapper8_copies81 a h).2.2.1
  have c2595 := (privateBatchWrapper8_copies81 a h).2.2.2.1
  have c2596 := (privateBatchWrapper8_copies81 a h).2.2.2.2.1
  have c2597 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.1
  have c2598 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.1
  have c2599 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.1
  have c2600 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.1
  have c2601 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.1
  have c2602 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2603 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2604 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2605 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2606 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2607 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2608 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2609 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2610 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2611 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2612 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2613 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2614 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2615 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2616 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2617 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2618 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2619 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2620 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2621 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2622 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2623 := (privateBatchWrapper8_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2624 := (privateBatchWrapper8_copies82 a h).1
  have c2625 := (privateBatchWrapper8_copies82 a h).2.1
  have c2626 := (privateBatchWrapper8_copies82 a h).2.2.1
  have c2627 := (privateBatchWrapper8_copies82 a h).2.2.2.1
  have c2628 := (privateBatchWrapper8_copies82 a h).2.2.2.2.1
  have c2629 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.1
  have c2630 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.1
  have c2631 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.1
  have c2632 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.1
  have c2633 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.1
  have c2634 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2635 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2636 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 69) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2591.trans k2
    · exact c2592.trans k2
    · exact c2593.trans k2
    · exact c2594.trans k2
    · exact c2595.trans k2
    · exact c2596.trans k2
    · exact c2597.trans k2
    · exact c2598.trans k2
    · exact c2599.trans k2
    · exact c2600.trans k2
    · exact c2601.trans k2
    · exact c2602.trans k2
    · exact c2603.trans k2
    · exact c2604.trans k2
    · exact c2605.trans k2
    · exact c2606.trans k2
    · exact c2607.trans k2
    · exact c2608.trans k2
    · exact c2609.trans k2
    · exact c2610.trans k2
    · exact c2611.trans k2
    · exact c2612.trans k2
    · exact c2613.trans k2
    · exact c2614.trans k2
    · exact c2615.trans k2
    · exact c2616.trans k2
    · exact c2617.trans k2
    · exact c2618.trans k2
    · exact c2619.trans k2
    · exact c2620.trans k2
    · exact c2621.trans k2
    · exact c2622.trans k2
    · exact c2623.trans k2
    · exact c2624.trans k2
    · exact c2625.trans k2
    · exact c2626.trans k2
    · exact c2627.trans k2
    · exact c2628.trans k2
    · exact c2629.trans k2
    · exact c2630.trans k2
    · exact c2631.trans k2
    · exact c2632.trans k2
    · exact c2633.trans k2
    · exact c2634.trans k2
    · exact c2635.trans k2
    )
  rwa [c2636] at hr

theorem privateBatchWrapper8_f289 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 47) = a (.virt 412) - a (.virt 90) := by
  have c2637 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2638 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2639 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_11 := arithEq_of_rows h (row := 33) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c2637, k1, ← c2638, k3, ← c2639] at e_33_11
  have hr := e_33_11
  simp only [k1]
  linear_combination hr

theorem privateBatchWrapper8_f290 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.wire 33 47)) 14 := by
  have c2640 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2641 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2642 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2643 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2644 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2645 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2646 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2647 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2648 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2649 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2650 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2651 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2652 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2653 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2654 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2655 := (privateBatchWrapper8_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2656 := (privateBatchWrapper8_copies83 a h).1
  have c2657 := (privateBatchWrapper8_copies83 a h).2.1
  have c2658 := (privateBatchWrapper8_copies83 a h).2.2.1
  have c2659 := (privateBatchWrapper8_copies83 a h).2.2.2.1
  have c2660 := (privateBatchWrapper8_copies83 a h).2.2.2.2.1
  have c2661 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.1
  have c2662 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.1
  have c2663 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.1
  have c2664 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.1
  have c2665 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.1
  have c2666 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2667 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2668 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2669 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2670 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2671 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2672 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2673 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2674 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2675 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2676 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2677 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2678 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2679 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2680 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2681 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2682 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2683 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2684 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2685 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 70) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2640.trans k2
    · exact c2641.trans k2
    · exact c2642.trans k2
    · exact c2643.trans k2
    · exact c2644.trans k2
    · exact c2645.trans k2
    · exact c2646.trans k2
    · exact c2647.trans k2
    · exact c2648.trans k2
    · exact c2649.trans k2
    · exact c2650.trans k2
    · exact c2651.trans k2
    · exact c2652.trans k2
    · exact c2653.trans k2
    · exact c2654.trans k2
    · exact c2655.trans k2
    · exact c2656.trans k2
    · exact c2657.trans k2
    · exact c2658.trans k2
    · exact c2659.trans k2
    · exact c2660.trans k2
    · exact c2661.trans k2
    · exact c2662.trans k2
    · exact c2663.trans k2
    · exact c2664.trans k2
    · exact c2665.trans k2
    · exact c2666.trans k2
    · exact c2667.trans k2
    · exact c2668.trans k2
    · exact c2669.trans k2
    · exact c2670.trans k2
    · exact c2671.trans k2
    · exact c2672.trans k2
    · exact c2673.trans k2
    · exact c2674.trans k2
    · exact c2675.trans k2
    · exact c2676.trans k2
    · exact c2677.trans k2
    · exact c2678.trans k2
    · exact c2679.trans k2
    · exact c2680.trans k2
    · exact c2681.trans k2
    · exact c2682.trans k2
    · exact c2683.trans k2
    · exact c2684.trans k2
    )
  rwa [c2685] at hr

theorem privateBatchWrapper8_f291 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 3) = a (.wire 62 27) * a (.virt 412) := by
  have c2686 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2687 := (privateBatchWrapper8_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2688 := (privateBatchWrapper8_copies84 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_71_0 := arithEq_of_rows h (row := 71) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_0
  simp only [← c2686, ← c2687, k1, ← c2688] at e_71_0
  have hr := e_71_0
  simp only [k1]
  linear_combination hr

theorem privateBatchWrapper8_f292 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 7) = a (.wire 34 23) * a (.wire 33 47) := by
  have c2689 := (privateBatchWrapper8_copies84 a h).2.1
  have c2690 := (privateBatchWrapper8_copies84 a h).2.2.1
  have c2691 := (privateBatchWrapper8_copies84 a h).2.2.2.1
  have e_71_1 := arithEq_of_rows h (row := 71) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_1
  simp only [← c2689, ← c2690, ← c2691] at e_71_1
  have hr := e_71_1
  linear_combination hr

theorem privateBatchWrapper8_f293 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 33 51) = a (.wire 71 7) - a (.wire 71 3) := by
  have c2692 := (privateBatchWrapper8_copies84 a h).2.2.2.2.1
  have c2693 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.1
  have c2694 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_12 := arithEq_of_rows h (row := 33) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c2692, ← c2693, k3, ← c2694] at e_33_12
  have hr := e_33_12
  linear_combination hr

theorem privateBatchWrapper8_f294 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    rangeCheck (a (.wire 33 51)) 52 := by
  have c2695 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.1
  have c2696 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.1
  have c2697 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.1
  have c2698 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2699 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2700 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2701 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2702 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 72) (N := 59) (n := 52) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2695.trans k2
    · exact c2696.trans k2
    · exact c2697.trans k2
    · exact c2698.trans k2
    · exact c2699.trans k2
    · exact c2700.trans k2
    · exact c2701.trans k2
    )
  rwa [c2702] at hr

theorem privateBatchWrapper8_f295 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 19) = bnot (a (.wire 3 31)) := by
  have c520 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c520, k3, ← c521, k3, ← c522] at e_12_4
  have hr := e_12_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f296 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 43) = bnot (a (.wire 4 15)) := by
  have c538 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c538, k3, ← c539, k3, ← c540] at e_12_10
  have hr := e_12_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f297 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 11) = band (a (.wire 12 19)) (a (.wire 12 43)) := by
  have c2703 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2704 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2705 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_2 := arithEq_of_rows h (row := 71) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_2
  simp only [← c2703, ← c2704, ← c2705] at e_71_2
  have hr := e_71_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f298 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 14)) (a (.virt 413)) (a (.virt 414)) := by
  have c2706 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2707 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2708 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2709 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2710 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2711 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2712 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2713 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2714 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2715 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2716 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2717 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2718 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2719 := (privateBatchWrapper8_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2720 := (privateBatchWrapper8_copies85 a h).1
  have c2721 := (privateBatchWrapper8_copies85 a h).2.1
  have c2722 := (privateBatchWrapper8_copies85 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_13 := arithEq_of_rows h (row := 33) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c2706, k3, ← c2707, k3, ← c2708] at e_33_13
  have e_33_14 := arithEq_of_rows h (row := 33) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c2709, ← c2710, k3, ← c2711] at e_33_14
  have e_71_3 := arithEq_of_rows h (row := 71) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_3
  simp only [← c2712, ← c2713, ← c2714] at e_71_3
  have e_71_4 := arithEq_of_rows h (row := 71) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_4
  simp only [← c2715, ← c2716, ← c2717] at e_71_4
  have e_73_0 := arithEq_of_rows h (row := 73) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_0
  simp only [← c2718, ← c2719, k3, ← c2720] at e_73_0
  refine ⟨?_, ?_⟩
  · have hc := e_71_3
    simp only [e_33_14] at hc
    linear_combination c2721.trans k2 - hc
  · have hc := e_73_0
    simp only [e_33_13, e_71_4, e_33_14] at hc
    linear_combination c2722.trans k2 - hc

theorem privateBatchWrapper8_f299 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 15)) (a (.virt 415)) (a (.virt 416)) := by
  have c2723 := (privateBatchWrapper8_copies85 a h).2.2.2.1
  have c2724 := (privateBatchWrapper8_copies85 a h).2.2.2.2.1
  have c2725 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.1
  have c2726 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.1
  have c2727 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.1
  have c2728 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.1
  have c2729 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.1
  have c2730 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2731 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2732 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2733 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2734 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2735 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2736 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2737 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2738 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2739 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_71_5 := arithEq_of_rows h (row := 71) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_5
  simp only [← c2729, ← c2730, ← c2731] at e_71_5
  have e_71_6 := arithEq_of_rows h (row := 71) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_6
  simp only [← c2732, ← c2733, ← c2734] at e_71_6
  have e_73_1 := arithEq_of_rows h (row := 73) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_1
  simp only [← c2723, k3, ← c2724, k3, ← c2725] at e_73_1
  have e_73_2 := arithEq_of_rows h (row := 73) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_2
  simp only [← c2726, ← c2727, k3, ← c2728] at e_73_2
  have e_73_3 := arithEq_of_rows h (row := 73) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_3
  simp only [← c2735, ← c2736, k3, ← c2737] at e_73_3
  refine ⟨?_, ?_⟩
  · have hc := e_71_5
    simp only [e_73_2] at hc
    linear_combination c2738.trans k2 - hc
  · have hc := e_73_3
    simp only [e_73_1, e_71_6, e_73_2] at hc
    linear_combination c2739.trans k2 - hc

theorem privateBatchWrapper8_f300 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 16)) (a (.virt 417)) (a (.virt 418)) := by
  have c2740 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2741 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2742 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2743 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2744 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2745 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2746 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2747 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2748 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2749 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2750 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2751 := (privateBatchWrapper8_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2752 := (privateBatchWrapper8_copies86 a h).1
  have c2753 := (privateBatchWrapper8_copies86 a h).2.1
  have c2754 := (privateBatchWrapper8_copies86 a h).2.2.1
  have c2755 := (privateBatchWrapper8_copies86 a h).2.2.2.1
  have c2756 := (privateBatchWrapper8_copies86 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_71_7 := arithEq_of_rows h (row := 71) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_7
  simp only [← c2746, ← c2747, ← c2748] at e_71_7
  have e_71_8 := arithEq_of_rows h (row := 71) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_8
  simp only [← c2749, ← c2750, ← c2751] at e_71_8
  have e_73_4 := arithEq_of_rows h (row := 73) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c2740, k3, ← c2741, k3, ← c2742] at e_73_4
  have e_73_5 := arithEq_of_rows h (row := 73) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_5
  simp only [← c2743, ← c2744, k3, ← c2745] at e_73_5
  have e_73_6 := arithEq_of_rows h (row := 73) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_6
  simp only [← c2752, ← c2753, k3, ← c2754] at e_73_6
  refine ⟨?_, ?_⟩
  · have hc := e_71_7
    simp only [e_73_5] at hc
    linear_combination c2755.trans k2 - hc
  · have hc := e_73_6
    simp only [e_73_4, e_71_8, e_73_5] at hc
    linear_combination c2756.trans k2 - hc

theorem privateBatchWrapper8_f301 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 17)) (a (.virt 419)) (a (.virt 420)) := by
  have c2757 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.1
  have c2758 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.1
  have c2759 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.1
  have c2760 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.1
  have c2761 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.1
  have c2762 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2763 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2764 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2765 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2766 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2767 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2768 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2769 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2770 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2771 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2772 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2773 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_71_9 := arithEq_of_rows h (row := 71) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_9
  simp only [← c2763, ← c2764, ← c2765] at e_71_9
  have e_71_10 := arithEq_of_rows h (row := 71) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_10
  simp only [← c2766, ← c2767, ← c2768] at e_71_10
  have e_73_7 := arithEq_of_rows h (row := 73) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_7
  simp only [← c2757, k3, ← c2758, k3, ← c2759] at e_73_7
  have e_73_8 := arithEq_of_rows h (row := 73) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_8
  simp only [← c2760, ← c2761, k3, ← c2762] at e_73_8
  have e_73_9 := arithEq_of_rows h (row := 73) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_9
  simp only [← c2769, ← c2770, k3, ← c2771] at e_73_9
  refine ⟨?_, ?_⟩
  · have hc := e_71_9
    simp only [e_73_8] at hc
    linear_combination c2772.trans k2 - hc
  · have hc := e_73_9
    simp only [e_73_7, e_71_10, e_73_8] at hc
    linear_combination c2773.trans k2 - hc

theorem privateBatchWrapper8_f302 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 47) = band (a (.virt 413)) (a (.virt 415)) := by
  have c2774 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2775 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2776 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_11 := arithEq_of_rows h (row := 71) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_11
  simp only [← c2774, ← c2775, ← c2776] at e_71_11
  have hr := e_71_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f303 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 51) = band (a (.virt 417)) (a (.virt 419)) := by
  have c2777 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2778 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2779 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_12 := arithEq_of_rows h (row := 71) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_12
  simp only [← c2777, ← c2778, ← c2779] at e_71_12
  have hr := e_71_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f304 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 55) = band (a (.wire 71 47)) (a (.wire 71 51)) := by
  have c2780 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2781 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2782 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_13 := arithEq_of_rows h (row := 71) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_13
  simp only [← c2780, ← c2781, ← c2782] at e_71_13
  have hr := e_71_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f305 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 59) = band (a (.wire 71 11)) (a (.wire 71 55)) := by
  have c2783 := (privateBatchWrapper8_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2784 := (privateBatchWrapper8_copies87 a h).1
  have c2785 := (privateBatchWrapper8_copies87 a h).2.1
  have e_71_14 := arithEq_of_rows h (row := 71) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_14
  simp only [← c2783, ← c2784, ← c2785] at e_71_14
  have hr := e_71_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f306 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 71 59) = a (.virt 11) := by
  have c2786 := (privateBatchWrapper8_copies87 a h).2.2.1
  exact c2786

theorem privateBatchWrapper8_f307 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 31) = bnot (a (.wire 4 59)) := by
  have c583 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c583, k3, ← c584, k3, ← c585] at e_13_7
  have hr := e_13_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f308 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 3) = band (a (.wire 12 19)) (a (.wire 13 31)) := by
  have c2787 := (privateBatchWrapper8_copies87 a h).2.2.2.1
  have c2788 := (privateBatchWrapper8_copies87 a h).2.2.2.2.1
  have c2789 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.1
  have e_74_0 := arithEq_of_rows h (row := 74) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_0
  simp only [← c2787, ← c2788, ← c2789] at e_74_0
  have hr := e_74_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f309 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 25)) (a (.virt 421)) (a (.virt 422)) := by
  have c2790 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.1
  have c2791 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.1
  have c2792 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.1
  have c2793 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.1
  have c2794 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2795 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2796 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2797 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2798 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2799 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2800 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2801 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2802 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2803 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2804 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2805 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2806 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_73_10 := arithEq_of_rows h (row := 73) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_10
  simp only [← c2790, k3, ← c2791, k3, ← c2792] at e_73_10
  have e_73_11 := arithEq_of_rows h (row := 73) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_11
  simp only [← c2793, ← c2794, k3, ← c2795] at e_73_11
  have e_73_12 := arithEq_of_rows h (row := 73) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_12
  simp only [← c2802, ← c2803, k3, ← c2804] at e_73_12
  have e_74_1 := arithEq_of_rows h (row := 74) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_1
  simp only [← c2796, ← c2797, ← c2798] at e_74_1
  have e_74_2 := arithEq_of_rows h (row := 74) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_2
  simp only [← c2799, ← c2800, ← c2801] at e_74_2
  refine ⟨?_, ?_⟩
  · have hc := e_74_1
    simp only [e_73_11] at hc
    linear_combination c2805.trans k2 - hc
  · have hc := e_73_12
    simp only [e_73_10, e_74_2, e_73_11] at hc
    linear_combination c2806.trans k2 - hc

theorem privateBatchWrapper8_f310 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 26)) (a (.virt 423)) (a (.virt 424)) := by
  have c2807 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2808 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2809 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2810 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2811 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2812 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2813 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2814 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2815 := (privateBatchWrapper8_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2816 := (privateBatchWrapper8_copies88 a h).1
  have c2817 := (privateBatchWrapper8_copies88 a h).2.1
  have c2818 := (privateBatchWrapper8_copies88 a h).2.2.1
  have c2819 := (privateBatchWrapper8_copies88 a h).2.2.2.1
  have c2820 := (privateBatchWrapper8_copies88 a h).2.2.2.2.1
  have c2821 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.1
  have c2822 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.1
  have c2823 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_73_13 := arithEq_of_rows h (row := 73) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_13
  simp only [← c2807, k3, ← c2808, k3, ← c2809] at e_73_13
  have e_73_14 := arithEq_of_rows h (row := 73) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_14
  simp only [← c2810, ← c2811, k3, ← c2812] at e_73_14
  have e_74_3 := arithEq_of_rows h (row := 74) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_3
  simp only [← c2813, ← c2814, ← c2815] at e_74_3
  have e_74_4 := arithEq_of_rows h (row := 74) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_4
  simp only [← c2816, ← c2817, ← c2818] at e_74_4
  have e_75_0 := arithEq_of_rows h (row := 75) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_0
  simp only [← c2819, ← c2820, k3, ← c2821] at e_75_0
  refine ⟨?_, ?_⟩
  · have hc := e_74_3
    simp only [e_73_14] at hc
    linear_combination c2822.trans k2 - hc
  · have hc := e_75_0
    simp only [e_73_13, e_74_4, e_73_14] at hc
    linear_combination c2823.trans k2 - hc

theorem privateBatchWrapper8_f311 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 27)) (a (.virt 425)) (a (.virt 426)) := by
  have c2824 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.1
  have c2825 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.1
  have c2826 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2827 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2828 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2829 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2830 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2831 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2832 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2833 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2834 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2835 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2836 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2837 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2838 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2839 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2840 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_74_5 := arithEq_of_rows h (row := 74) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_5
  simp only [← c2830, ← c2831, ← c2832] at e_74_5
  have e_74_6 := arithEq_of_rows h (row := 74) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_6
  simp only [← c2833, ← c2834, ← c2835] at e_74_6
  have e_75_1 := arithEq_of_rows h (row := 75) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_1
  simp only [← c2824, k3, ← c2825, k3, ← c2826] at e_75_1
  have e_75_2 := arithEq_of_rows h (row := 75) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_2
  simp only [← c2827, ← c2828, k3, ← c2829] at e_75_2
  have e_75_3 := arithEq_of_rows h (row := 75) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_3
  simp only [← c2836, ← c2837, k3, ← c2838] at e_75_3
  refine ⟨?_, ?_⟩
  · have hc := e_74_5
    simp only [e_75_2] at hc
    linear_combination c2839.trans k2 - hc
  · have hc := e_75_3
    simp only [e_75_1, e_74_6, e_75_2] at hc
    linear_combination c2840.trans k2 - hc

theorem privateBatchWrapper8_f312 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 28)) (a (.virt 427)) (a (.virt 428)) := by
  have c2841 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2842 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2843 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2844 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2845 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2846 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2847 := (privateBatchWrapper8_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2848 := (privateBatchWrapper8_copies89 a h).1
  have c2849 := (privateBatchWrapper8_copies89 a h).2.1
  have c2850 := (privateBatchWrapper8_copies89 a h).2.2.1
  have c2851 := (privateBatchWrapper8_copies89 a h).2.2.2.1
  have c2852 := (privateBatchWrapper8_copies89 a h).2.2.2.2.1
  have c2853 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.1
  have c2854 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.1
  have c2855 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.1
  have c2856 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.1
  have c2857 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_74_7 := arithEq_of_rows h (row := 74) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_7
  simp only [← c2847, ← c2848, ← c2849] at e_74_7
  have e_74_8 := arithEq_of_rows h (row := 74) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_8
  simp only [← c2850, ← c2851, ← c2852] at e_74_8
  have e_75_4 := arithEq_of_rows h (row := 75) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_4
  simp only [← c2841, k3, ← c2842, k3, ← c2843] at e_75_4
  have e_75_5 := arithEq_of_rows h (row := 75) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_5
  simp only [← c2844, ← c2845, k3, ← c2846] at e_75_5
  have e_75_6 := arithEq_of_rows h (row := 75) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_6
  simp only [← c2853, ← c2854, k3, ← c2855] at e_75_6
  refine ⟨?_, ?_⟩
  · have hc := e_74_7
    simp only [e_75_5] at hc
    linear_combination c2856.trans k2 - hc
  · have hc := e_75_6
    simp only [e_75_4, e_74_8, e_75_5] at hc
    linear_combination c2857.trans k2 - hc

theorem privateBatchWrapper8_f313 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 39) = band (a (.virt 421)) (a (.virt 423)) := by
  have c2858 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2859 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2860 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_9 := arithEq_of_rows h (row := 74) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_9
  simp only [← c2858, ← c2859, ← c2860] at e_74_9
  have hr := e_74_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f314 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 43) = band (a (.virt 425)) (a (.virt 427)) := by
  have c2861 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2862 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2863 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_10 := arithEq_of_rows h (row := 74) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_10
  simp only [← c2861, ← c2862, ← c2863] at e_74_10
  have hr := e_74_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f315 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 47) = band (a (.wire 74 39)) (a (.wire 74 43)) := by
  have c2864 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2865 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2866 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_11 := arithEq_of_rows h (row := 74) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_11
  simp only [← c2864, ← c2865, ← c2866] at e_74_11
  have hr := e_74_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f316 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 51) = band (a (.wire 74 3)) (a (.wire 74 47)) := by
  have c2867 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2868 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2869 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_12 := arithEq_of_rows h (row := 74) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_12
  simp only [← c2867, ← c2868, ← c2869] at e_74_12
  have hr := e_74_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f317 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 51) = a (.virt 11) := by
  have c2870 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c2870

theorem privateBatchWrapper8_f318 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 19) = bnot (a (.wire 6 43)) := by
  have c628 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c628, k3, ← c629, k3, ← c630] at e_15_4
  have hr := e_15_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f319 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 74 55) = band (a (.wire 12 19)) (a (.wire 15 19)) := by
  have c2871 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2872 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2873 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_13 := arithEq_of_rows h (row := 74) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_13
  simp only [← c2871, ← c2872, ← c2873] at e_74_13
  have hr := e_74_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f320 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 36)) (a (.virt 429)) (a (.virt 430)) := by
  have c2874 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2875 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2876 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2877 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2878 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2879 := (privateBatchWrapper8_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2880 := (privateBatchWrapper8_copies90 a h).1
  have c2881 := (privateBatchWrapper8_copies90 a h).2.1
  have c2882 := (privateBatchWrapper8_copies90 a h).2.2.1
  have c2883 := (privateBatchWrapper8_copies90 a h).2.2.2.1
  have c2884 := (privateBatchWrapper8_copies90 a h).2.2.2.2.1
  have c2885 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.1
  have c2886 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.1
  have c2887 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.1
  have c2888 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.1
  have c2889 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.1
  have c2890 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_74_14 := arithEq_of_rows h (row := 74) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_14
  simp only [← c2880, ← c2881, ← c2882] at e_74_14
  have e_75_7 := arithEq_of_rows h (row := 75) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_7
  simp only [← c2874, k3, ← c2875, k3, ← c2876] at e_75_7
  have e_75_8 := arithEq_of_rows h (row := 75) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_8
  simp only [← c2877, ← c2878, k3, ← c2879] at e_75_8
  have e_75_9 := arithEq_of_rows h (row := 75) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_9
  simp only [← c2886, ← c2887, k3, ← c2888] at e_75_9
  have e_76_0 := arithEq_of_rows h (row := 76) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_0
  simp only [← c2883, ← c2884, ← c2885] at e_76_0
  refine ⟨?_, ?_⟩
  · have hc := e_74_14
    simp only [e_75_8] at hc
    linear_combination c2889.trans k2 - hc
  · have hc := e_75_9
    simp only [e_75_7, e_76_0, e_75_8] at hc
    linear_combination c2890.trans k2 - hc

theorem privateBatchWrapper8_f321 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 37)) (a (.virt 431)) (a (.virt 432)) := by
  have c2891 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2892 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2893 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2894 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2895 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2896 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2897 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2898 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2899 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2900 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2901 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2902 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2903 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2904 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2905 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2906 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2907 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_75_10 := arithEq_of_rows h (row := 75) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_10
  simp only [← c2891, k3, ← c2892, k3, ← c2893] at e_75_10
  have e_75_11 := arithEq_of_rows h (row := 75) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_11
  simp only [← c2894, ← c2895, k3, ← c2896] at e_75_11
  have e_75_12 := arithEq_of_rows h (row := 75) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_12
  simp only [← c2903, ← c2904, k3, ← c2905] at e_75_12
  have e_76_1 := arithEq_of_rows h (row := 76) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_1
  simp only [← c2897, ← c2898, ← c2899] at e_76_1
  have e_76_2 := arithEq_of_rows h (row := 76) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_2
  simp only [← c2900, ← c2901, ← c2902] at e_76_2
  refine ⟨?_, ?_⟩
  · have hc := e_76_1
    simp only [e_75_11] at hc
    linear_combination c2906.trans k2 - hc
  · have hc := e_75_12
    simp only [e_75_10, e_76_2, e_75_11] at hc
    linear_combination c2907.trans k2 - hc

theorem privateBatchWrapper8_f322 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 38)) (a (.virt 433)) (a (.virt 434)) := by
  have c2908 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2909 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2910 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2911 := (privateBatchWrapper8_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2912 := (privateBatchWrapper8_copies91 a h).1
  have c2913 := (privateBatchWrapper8_copies91 a h).2.1
  have c2914 := (privateBatchWrapper8_copies91 a h).2.2.1
  have c2915 := (privateBatchWrapper8_copies91 a h).2.2.2.1
  have c2916 := (privateBatchWrapper8_copies91 a h).2.2.2.2.1
  have c2917 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.1
  have c2918 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.1
  have c2919 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.1
  have c2920 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.1
  have c2921 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.1
  have c2922 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2923 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2924 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_75_13 := arithEq_of_rows h (row := 75) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_13
  simp only [← c2908, k3, ← c2909, k3, ← c2910] at e_75_13
  have e_75_14 := arithEq_of_rows h (row := 75) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_14
  simp only [← c2911, ← c2912, k3, ← c2913] at e_75_14
  have e_76_3 := arithEq_of_rows h (row := 76) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_3
  simp only [← c2914, ← c2915, ← c2916] at e_76_3
  have e_76_4 := arithEq_of_rows h (row := 76) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_4
  simp only [← c2917, ← c2918, ← c2919] at e_76_4
  have e_77_0 := arithEq_of_rows h (row := 77) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_0
  simp only [← c2920, ← c2921, k3, ← c2922] at e_77_0
  refine ⟨?_, ?_⟩
  · have hc := e_76_3
    simp only [e_75_14] at hc
    linear_combination c2923.trans k2 - hc
  · have hc := e_77_0
    simp only [e_75_13, e_76_4, e_75_14] at hc
    linear_combination c2924.trans k2 - hc

theorem privateBatchWrapper8_f323 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 39)) (a (.virt 435)) (a (.virt 436)) := by
  have c2925 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2926 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2927 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2928 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2929 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2930 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2931 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2932 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2933 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2934 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2935 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2936 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2937 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2938 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2939 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2940 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2941 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_76_5 := arithEq_of_rows h (row := 76) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_5
  simp only [← c2931, ← c2932, ← c2933] at e_76_5
  have e_76_6 := arithEq_of_rows h (row := 76) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_6
  simp only [← c2934, ← c2935, ← c2936] at e_76_6
  have e_77_1 := arithEq_of_rows h (row := 77) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_1
  simp only [← c2925, k3, ← c2926, k3, ← c2927] at e_77_1
  have e_77_2 := arithEq_of_rows h (row := 77) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_2
  simp only [← c2928, ← c2929, k3, ← c2930] at e_77_2
  have e_77_3 := arithEq_of_rows h (row := 77) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_3
  simp only [← c2937, ← c2938, k3, ← c2939] at e_77_3
  refine ⟨?_, ?_⟩
  · have hc := e_76_5
    simp only [e_77_2] at hc
    linear_combination c2940.trans k2 - hc
  · have hc := e_77_3
    simp only [e_77_1, e_76_6, e_77_2] at hc
    linear_combination c2941.trans k2 - hc

theorem privateBatchWrapper8_f324 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 76 31) = band (a (.virt 429)) (a (.virt 431)) := by
  have c2942 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2943 := (privateBatchWrapper8_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2944 := (privateBatchWrapper8_copies92 a h).1
  have e_76_7 := arithEq_of_rows h (row := 76) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_7
  simp only [← c2942, ← c2943, ← c2944] at e_76_7
  have hr := e_76_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f325 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 76 35) = band (a (.virt 433)) (a (.virt 435)) := by
  have c2945 := (privateBatchWrapper8_copies92 a h).2.1
  have c2946 := (privateBatchWrapper8_copies92 a h).2.2.1
  have c2947 := (privateBatchWrapper8_copies92 a h).2.2.2.1
  have e_76_8 := arithEq_of_rows h (row := 76) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_8
  simp only [← c2945, ← c2946, ← c2947] at e_76_8
  have hr := e_76_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f326 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 76 39) = band (a (.wire 76 31)) (a (.wire 76 35)) := by
  have c2948 := (privateBatchWrapper8_copies92 a h).2.2.2.2.1
  have c2949 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.1
  have c2950 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.1
  have e_76_9 := arithEq_of_rows h (row := 76) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_9
  simp only [← c2948, ← c2949, ← c2950] at e_76_9
  have hr := e_76_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f327 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 76 43) = band (a (.wire 74 55)) (a (.wire 76 39)) := by
  have c2951 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.1
  have c2952 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.1
  have c2953 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.1
  have e_76_10 := arithEq_of_rows h (row := 76) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_10
  simp only [← c2951, ← c2952, ← c2953] at e_76_10
  have hr := e_76_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f328 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 76 43) = a (.virt 11) := by
  have c2954 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c2954

theorem privateBatchWrapper8_f329 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 7) = bnot (a (.wire 8 27)) := by
  have c673 := (privateBatchWrapper8_copies21 a h).2.1
  have c674 := (privateBatchWrapper8_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper8_copies21 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c673, k3, ← c674, k3, ← c675] at e_17_1
  have hr := e_17_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f330 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 76 47) = band (a (.wire 12 19)) (a (.wire 17 7)) := by
  have c2955 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2956 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2957 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_76_11 := arithEq_of_rows h (row := 76) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_11
  simp only [← c2955, ← c2956, ← c2957] at e_76_11
  have hr := e_76_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f331 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 47)) (a (.virt 437)) (a (.virt 438)) := by
  have c2958 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2959 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2960 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2961 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2962 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2963 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2964 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2965 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2966 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2967 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2968 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2969 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2970 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2971 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2972 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2973 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2974 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_76_12 := arithEq_of_rows h (row := 76) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_12
  simp only [← c2964, ← c2965, ← c2966] at e_76_12
  have e_76_13 := arithEq_of_rows h (row := 76) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_13
  simp only [← c2967, ← c2968, ← c2969] at e_76_13
  have e_77_4 := arithEq_of_rows h (row := 77) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_4
  simp only [← c2958, k3, ← c2959, k3, ← c2960] at e_77_4
  have e_77_5 := arithEq_of_rows h (row := 77) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_5
  simp only [← c2961, ← c2962, k3, ← c2963] at e_77_5
  have e_77_6 := arithEq_of_rows h (row := 77) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_6
  simp only [← c2970, ← c2971, k3, ← c2972] at e_77_6
  refine ⟨?_, ?_⟩
  · have hc := e_76_12
    simp only [e_77_5] at hc
    linear_combination c2973.trans k2 - hc
  · have hc := e_77_6
    simp only [e_77_4, e_76_13, e_77_5] at hc
    linear_combination c2974.trans k2 - hc

theorem privateBatchWrapper8_f332 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 48)) (a (.virt 439)) (a (.virt 440)) := by
  have c2975 := (privateBatchWrapper8_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2976 := (privateBatchWrapper8_copies93 a h).1
  have c2977 := (privateBatchWrapper8_copies93 a h).2.1
  have c2978 := (privateBatchWrapper8_copies93 a h).2.2.1
  have c2979 := (privateBatchWrapper8_copies93 a h).2.2.2.1
  have c2980 := (privateBatchWrapper8_copies93 a h).2.2.2.2.1
  have c2981 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.1
  have c2982 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.1
  have c2983 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.1
  have c2984 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.1
  have c2985 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.1
  have c2986 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2987 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2988 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2989 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2990 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2991 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_76_14 := arithEq_of_rows h (row := 76) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_14
  simp only [← c2981, ← c2982, ← c2983] at e_76_14
  have e_77_7 := arithEq_of_rows h (row := 77) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_7
  simp only [← c2975, k3, ← c2976, k3, ← c2977] at e_77_7
  have e_77_8 := arithEq_of_rows h (row := 77) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_8
  simp only [← c2978, ← c2979, k3, ← c2980] at e_77_8
  have e_77_9 := arithEq_of_rows h (row := 77) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_9
  simp only [← c2987, ← c2988, k3, ← c2989] at e_77_9
  have e_78_0 := arithEq_of_rows h (row := 78) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_0
  simp only [← c2984, ← c2985, ← c2986] at e_78_0
  refine ⟨?_, ?_⟩
  · have hc := e_76_14
    simp only [e_77_8] at hc
    linear_combination c2990.trans k2 - hc
  · have hc := e_77_9
    simp only [e_77_7, e_78_0, e_77_8] at hc
    linear_combination c2991.trans k2 - hc

theorem privateBatchWrapper8_f333 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 49)) (a (.virt 441)) (a (.virt 442)) := by
  have c2992 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2993 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2994 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2995 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2996 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3002 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3003 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3004 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3005 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3006 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3007 := (privateBatchWrapper8_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3008 := (privateBatchWrapper8_copies94 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_77_10 := arithEq_of_rows h (row := 77) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_10
  simp only [← c2992, k3, ← c2993, k3, ← c2994] at e_77_10
  have e_77_11 := arithEq_of_rows h (row := 77) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_11
  simp only [← c2995, ← c2996, k3, ← c2997] at e_77_11
  have e_77_12 := arithEq_of_rows h (row := 77) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_12
  simp only [← c3004, ← c3005, k3, ← c3006] at e_77_12
  have e_78_1 := arithEq_of_rows h (row := 78) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_1
  simp only [← c2998, ← c2999, ← c3000] at e_78_1
  have e_78_2 := arithEq_of_rows h (row := 78) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_2
  simp only [← c3001, ← c3002, ← c3003] at e_78_2
  refine ⟨?_, ?_⟩
  · have hc := e_78_1
    simp only [e_77_11] at hc
    linear_combination c3007.trans k2 - hc
  · have hc := e_77_12
    simp only [e_77_10, e_78_2, e_77_11] at hc
    linear_combination c3008.trans k2 - hc

theorem privateBatchWrapper8_f334 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 50)) (a (.virt 443)) (a (.virt 444)) := by
  have c3009 := (privateBatchWrapper8_copies94 a h).2.1
  have c3010 := (privateBatchWrapper8_copies94 a h).2.2.1
  have c3011 := (privateBatchWrapper8_copies94 a h).2.2.2.1
  have c3012 := (privateBatchWrapper8_copies94 a h).2.2.2.2.1
  have c3013 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.1
  have c3014 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.1
  have c3015 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.1
  have c3016 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.1
  have c3017 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.1
  have c3018 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3019 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3020 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3021 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3022 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3023 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3024 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3025 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_77_13 := arithEq_of_rows h (row := 77) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_13
  simp only [← c3009, k3, ← c3010, k3, ← c3011] at e_77_13
  have e_77_14 := arithEq_of_rows h (row := 77) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_14
  simp only [← c3012, ← c3013, k3, ← c3014] at e_77_14
  have e_78_3 := arithEq_of_rows h (row := 78) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_3
  simp only [← c3015, ← c3016, ← c3017] at e_78_3
  have e_78_4 := arithEq_of_rows h (row := 78) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_4
  simp only [← c3018, ← c3019, ← c3020] at e_78_4
  have e_79_0 := arithEq_of_rows h (row := 79) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_0
  simp only [← c3021, ← c3022, k3, ← c3023] at e_79_0
  refine ⟨?_, ?_⟩
  · have hc := e_78_3
    simp only [e_77_14] at hc
    linear_combination c3024.trans k2 - hc
  · have hc := e_79_0
    simp only [e_77_13, e_78_4, e_77_14] at hc
    linear_combination c3025.trans k2 - hc

theorem privateBatchWrapper8_f335 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 78 23) = band (a (.virt 437)) (a (.virt 439)) := by
  have c3026 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3027 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3028 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_5 := arithEq_of_rows h (row := 78) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_5
  simp only [← c3026, ← c3027, ← c3028] at e_78_5
  have hr := e_78_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f336 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 78 27) = band (a (.virt 441)) (a (.virt 443)) := by
  have c3029 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3030 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3031 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_6 := arithEq_of_rows h (row := 78) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_6
  simp only [← c3029, ← c3030, ← c3031] at e_78_6
  have hr := e_78_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f337 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 78 31) = band (a (.wire 78 23)) (a (.wire 78 27)) := by
  have c3032 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3033 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3034 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_7 := arithEq_of_rows h (row := 78) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_7
  simp only [← c3032, ← c3033, ← c3034] at e_78_7
  have hr := e_78_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f338 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 78 35) = band (a (.wire 76 47)) (a (.wire 78 31)) := by
  have c3035 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3036 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3037 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_8 := arithEq_of_rows h (row := 78) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_8
  simp only [← c3035, ← c3036, ← c3037] at e_78_8
  have hr := e_78_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f339 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 78 35) = a (.virt 11) := by
  have c3038 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3038

theorem privateBatchWrapper8_f340 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f341 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 78 39) = band (a (.wire 12 19)) (a (.wire 17 55)) := by
  have c3039 := (privateBatchWrapper8_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3040 := (privateBatchWrapper8_copies95 a h).1
  have c3041 := (privateBatchWrapper8_copies95 a h).2.1
  have e_78_9 := arithEq_of_rows h (row := 78) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_9
  simp only [← c3039, ← c3040, ← c3041] at e_78_9
  have hr := e_78_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f342 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 58)) (a (.virt 445)) (a (.virt 446)) := by
  have c3042 := (privateBatchWrapper8_copies95 a h).2.2.1
  have c3043 := (privateBatchWrapper8_copies95 a h).2.2.2.1
  have c3044 := (privateBatchWrapper8_copies95 a h).2.2.2.2.1
  have c3045 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.1
  have c3046 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.1
  have c3047 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.1
  have c3048 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.1
  have c3049 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.1
  have c3050 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3051 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3052 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3053 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3054 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3055 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3056 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3057 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3058 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_78_10 := arithEq_of_rows h (row := 78) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_10
  simp only [← c3048, ← c3049, ← c3050] at e_78_10
  have e_78_11 := arithEq_of_rows h (row := 78) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_11
  simp only [← c3051, ← c3052, ← c3053] at e_78_11
  have e_79_1 := arithEq_of_rows h (row := 79) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_1
  simp only [← c3042, k3, ← c3043, k3, ← c3044] at e_79_1
  have e_79_2 := arithEq_of_rows h (row := 79) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_2
  simp only [← c3045, ← c3046, k3, ← c3047] at e_79_2
  have e_79_3 := arithEq_of_rows h (row := 79) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_3
  simp only [← c3054, ← c3055, k3, ← c3056] at e_79_3
  refine ⟨?_, ?_⟩
  · have hc := e_78_10
    simp only [e_79_2] at hc
    linear_combination c3057.trans k2 - hc
  · have hc := e_79_3
    simp only [e_79_1, e_78_11, e_79_2] at hc
    linear_combination c3058.trans k2 - hc

theorem privateBatchWrapper8_f343 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 59)) (a (.virt 447)) (a (.virt 448)) := by
  have c3059 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3060 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3061 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3062 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3063 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3064 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3065 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3066 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3067 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3068 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3069 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3070 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3071 := (privateBatchWrapper8_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3072 := (privateBatchWrapper8_copies96 a h).1
  have c3073 := (privateBatchWrapper8_copies96 a h).2.1
  have c3074 := (privateBatchWrapper8_copies96 a h).2.2.1
  have c3075 := (privateBatchWrapper8_copies96 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_78_12 := arithEq_of_rows h (row := 78) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_12
  simp only [← c3065, ← c3066, ← c3067] at e_78_12
  have e_78_13 := arithEq_of_rows h (row := 78) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_13
  simp only [← c3068, ← c3069, ← c3070] at e_78_13
  have e_79_4 := arithEq_of_rows h (row := 79) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_4
  simp only [← c3059, k3, ← c3060, k3, ← c3061] at e_79_4
  have e_79_5 := arithEq_of_rows h (row := 79) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_5
  simp only [← c3062, ← c3063, k3, ← c3064] at e_79_5
  have e_79_6 := arithEq_of_rows h (row := 79) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_6
  simp only [← c3071, ← c3072, k3, ← c3073] at e_79_6
  refine ⟨?_, ?_⟩
  · have hc := e_78_12
    simp only [e_79_5] at hc
    linear_combination c3074.trans k2 - hc
  · have hc := e_79_6
    simp only [e_79_4, e_78_13, e_79_5] at hc
    linear_combination c3075.trans k2 - hc

theorem privateBatchWrapper8_f344 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 60)) (a (.virt 449)) (a (.virt 450)) := by
  have c3076 := (privateBatchWrapper8_copies96 a h).2.2.2.2.1
  have c3077 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.1
  have c3078 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.1
  have c3079 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.1
  have c3080 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.1
  have c3081 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.1
  have c3082 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3083 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3084 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3085 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3086 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3087 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3088 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3089 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3090 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3091 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3092 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_78_14 := arithEq_of_rows h (row := 78) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_14
  simp only [← c3082, ← c3083, ← c3084] at e_78_14
  have e_79_7 := arithEq_of_rows h (row := 79) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_7
  simp only [← c3076, k3, ← c3077, k3, ← c3078] at e_79_7
  have e_79_8 := arithEq_of_rows h (row := 79) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_8
  simp only [← c3079, ← c3080, k3, ← c3081] at e_79_8
  have e_79_9 := arithEq_of_rows h (row := 79) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_9
  simp only [← c3088, ← c3089, k3, ← c3090] at e_79_9
  have e_80_0 := arithEq_of_rows h (row := 80) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_0
  simp only [← c3085, ← c3086, ← c3087] at e_80_0
  refine ⟨?_, ?_⟩
  · have hc := e_78_14
    simp only [e_79_8] at hc
    linear_combination c3091.trans k2 - hc
  · have hc := e_79_9
    simp only [e_79_7, e_80_0, e_79_8] at hc
    linear_combination c3092.trans k2 - hc

theorem privateBatchWrapper8_f345 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 61)) (a (.virt 451)) (a (.virt 452)) := by
  have c3093 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3094 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3095 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3096 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3097 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3098 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3099 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3100 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3101 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3102 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3103 := (privateBatchWrapper8_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3104 := (privateBatchWrapper8_copies97 a h).1
  have c3105 := (privateBatchWrapper8_copies97 a h).2.1
  have c3106 := (privateBatchWrapper8_copies97 a h).2.2.1
  have c3107 := (privateBatchWrapper8_copies97 a h).2.2.2.1
  have c3108 := (privateBatchWrapper8_copies97 a h).2.2.2.2.1
  have c3109 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_79_10 := arithEq_of_rows h (row := 79) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_10
  simp only [← c3093, k3, ← c3094, k3, ← c3095] at e_79_10
  have e_79_11 := arithEq_of_rows h (row := 79) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_11
  simp only [← c3096, ← c3097, k3, ← c3098] at e_79_11
  have e_79_12 := arithEq_of_rows h (row := 79) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_12
  simp only [← c3105, ← c3106, k3, ← c3107] at e_79_12
  have e_80_1 := arithEq_of_rows h (row := 80) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_1
  simp only [← c3099, ← c3100, ← c3101] at e_80_1
  have e_80_2 := arithEq_of_rows h (row := 80) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_2
  simp only [← c3102, ← c3103, ← c3104] at e_80_2
  refine ⟨?_, ?_⟩
  · have hc := e_80_1
    simp only [e_79_11] at hc
    linear_combination c3108.trans k2 - hc
  · have hc := e_79_12
    simp only [e_79_10, e_80_2, e_79_11] at hc
    linear_combination c3109.trans k2 - hc

theorem privateBatchWrapper8_f346 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 80 15) = band (a (.virt 445)) (a (.virt 447)) := by
  have c3110 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.1
  have c3111 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.1
  have c3112 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.1
  have e_80_3 := arithEq_of_rows h (row := 80) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_3
  simp only [← c3110, ← c3111, ← c3112] at e_80_3
  have hr := e_80_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f347 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 80 19) = band (a (.virt 449)) (a (.virt 451)) := by
  have c3113 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.1
  have c3114 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3115 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_4 := arithEq_of_rows h (row := 80) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_4
  simp only [← c3113, ← c3114, ← c3115] at e_80_4
  have hr := e_80_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f348 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 80 23) = band (a (.wire 80 15)) (a (.wire 80 19)) := by
  have c3116 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3117 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3118 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_5 := arithEq_of_rows h (row := 80) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_5
  simp only [← c3116, ← c3117, ← c3118] at e_80_5
  have hr := e_80_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f349 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 80 27) = band (a (.wire 78 39)) (a (.wire 80 23)) := by
  have c3119 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3120 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3121 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_6 := arithEq_of_rows h (row := 80) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_6
  simp only [← c3119, ← c3120, ← c3121] at e_80_6
  have hr := e_80_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f350 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 80 27) = a (.virt 11) := by
  have c3122 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3122

theorem privateBatchWrapper8_f351 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f352 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 80 31) = band (a (.wire 12 19)) (a (.wire 18 43)) := by
  have c3123 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3124 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3125 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_7 := arithEq_of_rows h (row := 80) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_7
  simp only [← c3123, ← c3124, ← c3125] at e_80_7
  have hr := e_80_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f353 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 69)) (a (.virt 453)) (a (.virt 454)) := by
  have c3126 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3127 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3128 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3129 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3130 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3131 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3132 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3133 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3134 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3135 := (privateBatchWrapper8_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3136 := (privateBatchWrapper8_copies98 a h).1
  have c3137 := (privateBatchWrapper8_copies98 a h).2.1
  have c3138 := (privateBatchWrapper8_copies98 a h).2.2.1
  have c3139 := (privateBatchWrapper8_copies98 a h).2.2.2.1
  have c3140 := (privateBatchWrapper8_copies98 a h).2.2.2.2.1
  have c3141 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.1
  have c3142 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_79_13 := arithEq_of_rows h (row := 79) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_13
  simp only [← c3126, k3, ← c3127, k3, ← c3128] at e_79_13
  have e_79_14 := arithEq_of_rows h (row := 79) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_14
  simp only [← c3129, ← c3130, k3, ← c3131] at e_79_14
  have e_80_8 := arithEq_of_rows h (row := 80) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_8
  simp only [← c3132, ← c3133, ← c3134] at e_80_8
  have e_80_9 := arithEq_of_rows h (row := 80) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_9
  simp only [← c3135, ← c3136, ← c3137] at e_80_9
  have e_81_0 := arithEq_of_rows h (row := 81) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_0
  simp only [← c3138, ← c3139, k3, ← c3140] at e_81_0
  refine ⟨?_, ?_⟩
  · have hc := e_80_8
    simp only [e_79_14] at hc
    linear_combination c3141.trans k2 - hc
  · have hc := e_81_0
    simp only [e_79_13, e_80_9, e_79_14] at hc
    linear_combination c3142.trans k2 - hc

theorem privateBatchWrapper8_f354 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 70)) (a (.virt 455)) (a (.virt 456)) := by
  have c3143 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.1
  have c3144 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.1
  have c3145 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.1
  have c3146 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3147 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3148 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3149 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3150 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3151 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3152 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3153 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3154 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3155 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3156 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3157 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3158 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3159 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_80_10 := arithEq_of_rows h (row := 80) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_10
  simp only [← c3149, ← c3150, ← c3151] at e_80_10
  have e_80_11 := arithEq_of_rows h (row := 80) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_11
  simp only [← c3152, ← c3153, ← c3154] at e_80_11
  have e_81_1 := arithEq_of_rows h (row := 81) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_1
  simp only [← c3143, k3, ← c3144, k3, ← c3145] at e_81_1
  have e_81_2 := arithEq_of_rows h (row := 81) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_2
  simp only [← c3146, ← c3147, k3, ← c3148] at e_81_2
  have e_81_3 := arithEq_of_rows h (row := 81) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_3
  simp only [← c3155, ← c3156, k3, ← c3157] at e_81_3
  refine ⟨?_, ?_⟩
  · have hc := e_80_10
    simp only [e_81_2] at hc
    linear_combination c3158.trans k2 - hc
  · have hc := e_81_3
    simp only [e_81_1, e_80_11, e_81_2] at hc
    linear_combination c3159.trans k2 - hc

theorem privateBatchWrapper8_f355 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 71)) (a (.virt 457)) (a (.virt 458)) := by
  have c3160 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3161 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3162 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3163 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3164 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3165 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3166 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3167 := (privateBatchWrapper8_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3168 := (privateBatchWrapper8_copies99 a h).1
  have c3169 := (privateBatchWrapper8_copies99 a h).2.1
  have c3170 := (privateBatchWrapper8_copies99 a h).2.2.1
  have c3171 := (privateBatchWrapper8_copies99 a h).2.2.2.1
  have c3172 := (privateBatchWrapper8_copies99 a h).2.2.2.2.1
  have c3173 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.1
  have c3174 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.1
  have c3175 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.1
  have c3176 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_80_12 := arithEq_of_rows h (row := 80) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_12
  simp only [← c3166, ← c3167, ← c3168] at e_80_12
  have e_80_13 := arithEq_of_rows h (row := 80) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_13
  simp only [← c3169, ← c3170, ← c3171] at e_80_13
  have e_81_4 := arithEq_of_rows h (row := 81) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_4
  simp only [← c3160, k3, ← c3161, k3, ← c3162] at e_81_4
  have e_81_5 := arithEq_of_rows h (row := 81) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_5
  simp only [← c3163, ← c3164, k3, ← c3165] at e_81_5
  have e_81_6 := arithEq_of_rows h (row := 81) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_6
  simp only [← c3172, ← c3173, k3, ← c3174] at e_81_6
  refine ⟨?_, ?_⟩
  · have hc := e_80_12
    simp only [e_81_5] at hc
    linear_combination c3175.trans k2 - hc
  · have hc := e_81_6
    simp only [e_81_4, e_80_13, e_81_5] at hc
    linear_combination c3176.trans k2 - hc

theorem privateBatchWrapper8_f356 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 72)) (a (.virt 459)) (a (.virt 460)) := by
  have c3177 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.1
  have c3178 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3179 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3180 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3181 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3182 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3183 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3184 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3185 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3186 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3187 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3188 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3189 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3190 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3191 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3192 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3193 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_80_14 := arithEq_of_rows h (row := 80) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_14
  simp only [← c3183, ← c3184, ← c3185] at e_80_14
  have e_81_7 := arithEq_of_rows h (row := 81) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_7
  simp only [← c3177, k3, ← c3178, k3, ← c3179] at e_81_7
  have e_81_8 := arithEq_of_rows h (row := 81) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_8
  simp only [← c3180, ← c3181, k3, ← c3182] at e_81_8
  have e_81_9 := arithEq_of_rows h (row := 81) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_9
  simp only [← c3189, ← c3190, k3, ← c3191] at e_81_9
  have e_82_0 := arithEq_of_rows h (row := 82) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_0
  simp only [← c3186, ← c3187, ← c3188] at e_82_0
  refine ⟨?_, ?_⟩
  · have hc := e_80_14
    simp only [e_81_8] at hc
    linear_combination c3192.trans k2 - hc
  · have hc := e_81_9
    simp only [e_81_7, e_82_0, e_81_8] at hc
    linear_combination c3193.trans k2 - hc

theorem privateBatchWrapper8_f357 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 7) = band (a (.virt 453)) (a (.virt 455)) := by
  have c3194 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3195 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3196 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_82_1 := arithEq_of_rows h (row := 82) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_1
  simp only [← c3194, ← c3195, ← c3196] at e_82_1
  have hr := e_82_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f358 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 11) = band (a (.virt 457)) (a (.virt 459)) := by
  have c3197 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3198 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3199 := (privateBatchWrapper8_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_82_2 := arithEq_of_rows h (row := 82) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_2
  simp only [← c3197, ← c3198, ← c3199] at e_82_2
  have hr := e_82_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f359 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 15) = band (a (.wire 82 7)) (a (.wire 82 11)) := by
  have c3200 := (privateBatchWrapper8_copies100 a h).1
  have c3201 := (privateBatchWrapper8_copies100 a h).2.1
  have c3202 := (privateBatchWrapper8_copies100 a h).2.2.1
  have e_82_3 := arithEq_of_rows h (row := 82) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_3
  simp only [← c3200, ← c3201, ← c3202] at e_82_3
  have hr := e_82_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f360 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 19) = band (a (.wire 80 31)) (a (.wire 82 15)) := by
  have c3203 := (privateBatchWrapper8_copies100 a h).2.2.2.1
  have c3204 := (privateBatchWrapper8_copies100 a h).2.2.2.2.1
  have c3205 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.1
  have e_82_4 := arithEq_of_rows h (row := 82) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_4
  simp only [← c3203, ← c3204, ← c3205] at e_82_4
  have hr := e_82_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f361 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 19) = a (.virt 11) := by
  have c3206 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.1
  exact c3206

theorem privateBatchWrapper8_f362 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f363 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 23) = band (a (.wire 12 19)) (a (.wire 20 31)) := by
  have c3207 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.1
  have c3208 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.1
  have c3209 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.1
  have e_82_5 := arithEq_of_rows h (row := 82) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_5
  simp only [← c3207, ← c3208, ← c3209] at e_82_5
  have hr := e_82_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f364 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 80)) (a (.virt 461)) (a (.virt 462)) := by
  have c3210 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3211 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3212 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3213 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3214 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3215 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3216 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3217 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3218 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3219 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3220 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3221 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3222 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3223 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3224 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3225 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3226 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_81_10 := arithEq_of_rows h (row := 81) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_10
  simp only [← c3210, k3, ← c3211, k3, ← c3212] at e_81_10
  have e_81_11 := arithEq_of_rows h (row := 81) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_11
  simp only [← c3213, ← c3214, k3, ← c3215] at e_81_11
  have e_81_12 := arithEq_of_rows h (row := 81) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_12
  simp only [← c3222, ← c3223, k3, ← c3224] at e_81_12
  have e_82_6 := arithEq_of_rows h (row := 82) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_6
  simp only [← c3216, ← c3217, ← c3218] at e_82_6
  have e_82_7 := arithEq_of_rows h (row := 82) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_7
  simp only [← c3219, ← c3220, ← c3221] at e_82_7
  refine ⟨?_, ?_⟩
  · have hc := e_82_6
    simp only [e_81_11] at hc
    linear_combination c3225.trans k2 - hc
  · have hc := e_81_12
    simp only [e_81_10, e_82_7, e_81_11] at hc
    linear_combination c3226.trans k2 - hc

theorem privateBatchWrapper8_f365 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 81)) (a (.virt 463)) (a (.virt 464)) := by
  have c3227 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3228 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3229 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3230 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3231 := (privateBatchWrapper8_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3232 := (privateBatchWrapper8_copies101 a h).1
  have c3233 := (privateBatchWrapper8_copies101 a h).2.1
  have c3234 := (privateBatchWrapper8_copies101 a h).2.2.1
  have c3235 := (privateBatchWrapper8_copies101 a h).2.2.2.1
  have c3236 := (privateBatchWrapper8_copies101 a h).2.2.2.2.1
  have c3237 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.1
  have c3238 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.1
  have c3239 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.1
  have c3240 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.1
  have c3241 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.1
  have c3242 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3243 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_81_13 := arithEq_of_rows h (row := 81) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_13
  simp only [← c3227, k3, ← c3228, k3, ← c3229] at e_81_13
  have e_81_14 := arithEq_of_rows h (row := 81) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_14
  simp only [← c3230, ← c3231, k3, ← c3232] at e_81_14
  have e_82_8 := arithEq_of_rows h (row := 82) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_8
  simp only [← c3233, ← c3234, ← c3235] at e_82_8
  have e_82_9 := arithEq_of_rows h (row := 82) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_9
  simp only [← c3236, ← c3237, ← c3238] at e_82_9
  have e_83_0 := arithEq_of_rows h (row := 83) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_0
  simp only [← c3239, ← c3240, k3, ← c3241] at e_83_0
  refine ⟨?_, ?_⟩
  · have hc := e_82_8
    simp only [e_81_14] at hc
    linear_combination c3242.trans k2 - hc
  · have hc := e_83_0
    simp only [e_81_13, e_82_9, e_81_14] at hc
    linear_combination c3243.trans k2 - hc

theorem privateBatchWrapper8_f366 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 82)) (a (.virt 465)) (a (.virt 466)) := by
  have c3244 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3245 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3246 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3247 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3248 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3249 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3250 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3251 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3252 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3253 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3254 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3255 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3256 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3257 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3258 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3259 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3260 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_82_10 := arithEq_of_rows h (row := 82) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_10
  simp only [← c3250, ← c3251, ← c3252] at e_82_10
  have e_82_11 := arithEq_of_rows h (row := 82) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_11
  simp only [← c3253, ← c3254, ← c3255] at e_82_11
  have e_83_1 := arithEq_of_rows h (row := 83) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_1
  simp only [← c3244, k3, ← c3245, k3, ← c3246] at e_83_1
  have e_83_2 := arithEq_of_rows h (row := 83) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_2
  simp only [← c3247, ← c3248, k3, ← c3249] at e_83_2
  have e_83_3 := arithEq_of_rows h (row := 83) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_3
  simp only [← c3256, ← c3257, k3, ← c3258] at e_83_3
  refine ⟨?_, ?_⟩
  · have hc := e_82_10
    simp only [e_83_2] at hc
    linear_combination c3259.trans k2 - hc
  · have hc := e_83_3
    simp only [e_83_1, e_82_11, e_83_2] at hc
    linear_combination c3260.trans k2 - hc

theorem privateBatchWrapper8_f367 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 83)) (a (.virt 467)) (a (.virt 468)) := by
  have c3261 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3262 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3263 := (privateBatchWrapper8_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3264 := (privateBatchWrapper8_copies102 a h).1
  have c3265 := (privateBatchWrapper8_copies102 a h).2.1
  have c3266 := (privateBatchWrapper8_copies102 a h).2.2.1
  have c3267 := (privateBatchWrapper8_copies102 a h).2.2.2.1
  have c3268 := (privateBatchWrapper8_copies102 a h).2.2.2.2.1
  have c3269 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.1
  have c3270 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.1
  have c3271 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.1
  have c3272 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.1
  have c3273 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.1
  have c3274 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3275 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3276 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3277 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_82_12 := arithEq_of_rows h (row := 82) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_12
  simp only [← c3267, ← c3268, ← c3269] at e_82_12
  have e_82_13 := arithEq_of_rows h (row := 82) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_13
  simp only [← c3270, ← c3271, ← c3272] at e_82_13
  have e_83_4 := arithEq_of_rows h (row := 83) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_4
  simp only [← c3261, k3, ← c3262, k3, ← c3263] at e_83_4
  have e_83_5 := arithEq_of_rows h (row := 83) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_5
  simp only [← c3264, ← c3265, k3, ← c3266] at e_83_5
  have e_83_6 := arithEq_of_rows h (row := 83) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_6
  simp only [← c3273, ← c3274, k3, ← c3275] at e_83_6
  refine ⟨?_, ?_⟩
  · have hc := e_82_12
    simp only [e_83_5] at hc
    linear_combination c3276.trans k2 - hc
  · have hc := e_83_6
    simp only [e_83_4, e_82_13, e_83_5] at hc
    linear_combination c3277.trans k2 - hc

theorem privateBatchWrapper8_f368 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 82 59) = band (a (.virt 461)) (a (.virt 463)) := by
  have c3278 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3279 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3280 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_82_14 := arithEq_of_rows h (row := 82) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_14
  simp only [← c3278, ← c3279, ← c3280] at e_82_14
  have hr := e_82_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f369 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 3) = band (a (.virt 465)) (a (.virt 467)) := by
  have c3281 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3282 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3283 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_0 := arithEq_of_rows h (row := 84) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_0
  simp only [← c3281, ← c3282, ← c3283] at e_84_0
  have hr := e_84_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f370 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 7) = band (a (.wire 82 59)) (a (.wire 84 3)) := by
  have c3284 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3285 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3286 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_1 := arithEq_of_rows h (row := 84) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_1
  simp only [← c3284, ← c3285, ← c3286] at e_84_1
  have hr := e_84_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f371 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 11) = band (a (.wire 82 23)) (a (.wire 84 7)) := by
  have c3287 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3288 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3289 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_2 := arithEq_of_rows h (row := 84) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_2
  simp only [← c3287, ← c3288, ← c3289] at e_84_2
  have hr := e_84_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f372 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 11) = a (.virt 11) := by
  have c3290 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3290

theorem privateBatchWrapper8_f373 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 12 43) = bnot (a (.wire 4 15)) := by
  have c538 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper8_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c538, k3, ← c539, k3, ← c540] at e_12_10
  have hr := e_12_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f374 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 31) = bnot (a (.wire 4 59)) := by
  have c583 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c583, k3, ← c584, k3, ← c585] at e_13_7
  have hr := e_13_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f375 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 15) = band (a (.wire 12 43)) (a (.wire 13 31)) := by
  have c3291 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3292 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3293 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_3 := arithEq_of_rows h (row := 84) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_3
  simp only [← c3291, ← c3292, ← c3293] at e_84_3
  have hr := e_84_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f376 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 14)) (a (.virt 25)) (a (.virt 469)) (a (.virt 470)) := by
  have c3294 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3295 := (privateBatchWrapper8_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3296 := (privateBatchWrapper8_copies103 a h).1
  have c3297 := (privateBatchWrapper8_copies103 a h).2.1
  have c3298 := (privateBatchWrapper8_copies103 a h).2.2.1
  have c3299 := (privateBatchWrapper8_copies103 a h).2.2.2.1
  have c3300 := (privateBatchWrapper8_copies103 a h).2.2.2.2.1
  have c3301 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.1
  have c3302 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.1
  have c3303 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.1
  have c3304 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.1
  have c3305 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.1
  have c3306 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3307 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3308 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3309 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3310 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_83_7 := arithEq_of_rows h (row := 83) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_7
  simp only [← c3294, k3, ← c3295, k3, ← c3296] at e_83_7
  have e_83_8 := arithEq_of_rows h (row := 83) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_8
  simp only [← c3297, ← c3298, k3, ← c3299] at e_83_8
  have e_83_9 := arithEq_of_rows h (row := 83) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_9
  simp only [← c3306, ← c3307, k3, ← c3308] at e_83_9
  have e_84_4 := arithEq_of_rows h (row := 84) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_4
  simp only [← c3300, ← c3301, ← c3302] at e_84_4
  have e_84_5 := arithEq_of_rows h (row := 84) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_5
  simp only [← c3303, ← c3304, ← c3305] at e_84_5
  refine ⟨?_, ?_⟩
  · have hc := e_84_4
    simp only [e_83_8] at hc
    linear_combination c3309.trans k2 - hc
  · have hc := e_83_9
    simp only [e_83_7, e_84_5, e_83_8] at hc
    linear_combination c3310.trans k2 - hc

theorem privateBatchWrapper8_f377 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 26)) (a (.virt 471)) (a (.virt 472)) := by
  have c3311 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3312 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3313 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3314 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3315 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3316 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3317 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3318 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3319 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3320 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3321 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3322 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3323 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3324 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3325 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3326 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3327 := (privateBatchWrapper8_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_83_10 := arithEq_of_rows h (row := 83) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_10
  simp only [← c3311, k3, ← c3312, k3, ← c3313] at e_83_10
  have e_83_11 := arithEq_of_rows h (row := 83) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_11
  simp only [← c3314, ← c3315, k3, ← c3316] at e_83_11
  have e_83_12 := arithEq_of_rows h (row := 83) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_12
  simp only [← c3323, ← c3324, k3, ← c3325] at e_83_12
  have e_84_6 := arithEq_of_rows h (row := 84) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_6
  simp only [← c3317, ← c3318, ← c3319] at e_84_6
  have e_84_7 := arithEq_of_rows h (row := 84) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_7
  simp only [← c3320, ← c3321, ← c3322] at e_84_7
  refine ⟨?_, ?_⟩
  · have hc := e_84_6
    simp only [e_83_11] at hc
    linear_combination c3326.trans k2 - hc
  · have hc := e_83_12
    simp only [e_83_10, e_84_7, e_83_11] at hc
    linear_combination c3327.trans k2 - hc

theorem privateBatchWrapper8_f378 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 27)) (a (.virt 473)) (a (.virt 474)) := by
  have c3328 := (privateBatchWrapper8_copies104 a h).1
  have c3329 := (privateBatchWrapper8_copies104 a h).2.1
  have c3330 := (privateBatchWrapper8_copies104 a h).2.2.1
  have c3331 := (privateBatchWrapper8_copies104 a h).2.2.2.1
  have c3332 := (privateBatchWrapper8_copies104 a h).2.2.2.2.1
  have c3333 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.1
  have c3334 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.1
  have c3335 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.1
  have c3336 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.1
  have c3337 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.1
  have c3338 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3339 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3340 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3341 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3342 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3343 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3344 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_83_13 := arithEq_of_rows h (row := 83) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_13
  simp only [← c3328, k3, ← c3329, k3, ← c3330] at e_83_13
  have e_83_14 := arithEq_of_rows h (row := 83) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_14
  simp only [← c3331, ← c3332, k3, ← c3333] at e_83_14
  have e_84_8 := arithEq_of_rows h (row := 84) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_8
  simp only [← c3334, ← c3335, ← c3336] at e_84_8
  have e_84_9 := arithEq_of_rows h (row := 84) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_9
  simp only [← c3337, ← c3338, ← c3339] at e_84_9
  have e_85_0 := arithEq_of_rows h (row := 85) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_0
  simp only [← c3340, ← c3341, k3, ← c3342] at e_85_0
  refine ⟨?_, ?_⟩
  · have hc := e_84_8
    simp only [e_83_14] at hc
    linear_combination c3343.trans k2 - hc
  · have hc := e_85_0
    simp only [e_83_13, e_84_9, e_83_14] at hc
    linear_combination c3344.trans k2 - hc

theorem privateBatchWrapper8_f379 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 28)) (a (.virt 475)) (a (.virt 476)) := by
  have c3345 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3346 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3347 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3348 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3349 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3350 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3351 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3352 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3353 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3354 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3355 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3356 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3357 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3358 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3359 := (privateBatchWrapper8_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3360 := (privateBatchWrapper8_copies105 a h).1
  have c3361 := (privateBatchWrapper8_copies105 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_84_10 := arithEq_of_rows h (row := 84) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_10
  simp only [← c3351, ← c3352, ← c3353] at e_84_10
  have e_84_11 := arithEq_of_rows h (row := 84) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_11
  simp only [← c3354, ← c3355, ← c3356] at e_84_11
  have e_85_1 := arithEq_of_rows h (row := 85) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_1
  simp only [← c3345, k3, ← c3346, k3, ← c3347] at e_85_1
  have e_85_2 := arithEq_of_rows h (row := 85) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_2
  simp only [← c3348, ← c3349, k3, ← c3350] at e_85_2
  have e_85_3 := arithEq_of_rows h (row := 85) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_3
  simp only [← c3357, ← c3358, k3, ← c3359] at e_85_3
  refine ⟨?_, ?_⟩
  · have hc := e_84_10
    simp only [e_85_2] at hc
    linear_combination c3360.trans k2 - hc
  · have hc := e_85_3
    simp only [e_85_1, e_84_11, e_85_2] at hc
    linear_combination c3361.trans k2 - hc

theorem privateBatchWrapper8_f380 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 51) = band (a (.virt 469)) (a (.virt 471)) := by
  have c3362 := (privateBatchWrapper8_copies105 a h).2.2.1
  have c3363 := (privateBatchWrapper8_copies105 a h).2.2.2.1
  have c3364 := (privateBatchWrapper8_copies105 a h).2.2.2.2.1
  have e_84_12 := arithEq_of_rows h (row := 84) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_12
  simp only [← c3362, ← c3363, ← c3364] at e_84_12
  have hr := e_84_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f381 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 55) = band (a (.virt 473)) (a (.virt 475)) := by
  have c3365 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.1
  have c3366 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.1
  have c3367 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.1
  have e_84_13 := arithEq_of_rows h (row := 84) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_13
  simp only [← c3365, ← c3366, ← c3367] at e_84_13
  have hr := e_84_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f382 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 84 59) = band (a (.wire 84 51)) (a (.wire 84 55)) := by
  have c3368 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.1
  have c3369 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.1
  have c3370 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_84_14 := arithEq_of_rows h (row := 84) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_14
  simp only [← c3368, ← c3369, ← c3370] at e_84_14
  have hr := e_84_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f383 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 3) = band (a (.wire 84 15)) (a (.wire 84 59)) := by
  have c3371 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3372 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3373 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_0 := arithEq_of_rows h (row := 86) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_0
  simp only [← c3371, ← c3372, ← c3373] at e_86_0
  have hr := e_86_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f384 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 3) = a (.virt 11) := by
  have c3374 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3374

theorem privateBatchWrapper8_f385 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 19) = bnot (a (.wire 6 43)) := by
  have c628 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c628, k3, ← c629, k3, ← c630] at e_15_4
  have hr := e_15_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f386 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 7) = band (a (.wire 12 43)) (a (.wire 15 19)) := by
  have c3375 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3376 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3377 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_1 := arithEq_of_rows h (row := 86) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_1
  simp only [← c3375, ← c3376, ← c3377] at e_86_1
  have hr := e_86_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f387 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 14)) (a (.virt 36)) (a (.virt 477)) (a (.virt 478)) := by
  have c3378 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3379 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3380 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3381 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3382 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3383 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3384 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3385 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3386 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3387 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3388 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3389 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3390 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3391 := (privateBatchWrapper8_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3392 := (privateBatchWrapper8_copies106 a h).1
  have c3393 := (privateBatchWrapper8_copies106 a h).2.1
  have c3394 := (privateBatchWrapper8_copies106 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_85_4 := arithEq_of_rows h (row := 85) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_4
  simp only [← c3378, k3, ← c3379, k3, ← c3380] at e_85_4
  have e_85_5 := arithEq_of_rows h (row := 85) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_5
  simp only [← c3381, ← c3382, k3, ← c3383] at e_85_5
  have e_85_6 := arithEq_of_rows h (row := 85) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_6
  simp only [← c3390, ← c3391, k3, ← c3392] at e_85_6
  have e_86_2 := arithEq_of_rows h (row := 86) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_2
  simp only [← c3384, ← c3385, ← c3386] at e_86_2
  have e_86_3 := arithEq_of_rows h (row := 86) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c3387, ← c3388, ← c3389] at e_86_3
  refine ⟨?_, ?_⟩
  · have hc := e_86_2
    simp only [e_85_5] at hc
    linear_combination c3393.trans k2 - hc
  · have hc := e_85_6
    simp only [e_85_4, e_86_3, e_85_5] at hc
    linear_combination c3394.trans k2 - hc

theorem privateBatchWrapper8_f388 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 37)) (a (.virt 479)) (a (.virt 480)) := by
  have c3395 := (privateBatchWrapper8_copies106 a h).2.2.2.1
  have c3396 := (privateBatchWrapper8_copies106 a h).2.2.2.2.1
  have c3397 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.1
  have c3398 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.1
  have c3399 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.1
  have c3400 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.1
  have c3401 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.1
  have c3402 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3403 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3404 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3405 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3406 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3407 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3408 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3409 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3410 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3411 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_85_7 := arithEq_of_rows h (row := 85) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_7
  simp only [← c3395, k3, ← c3396, k3, ← c3397] at e_85_7
  have e_85_8 := arithEq_of_rows h (row := 85) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_8
  simp only [← c3398, ← c3399, k3, ← c3400] at e_85_8
  have e_85_9 := arithEq_of_rows h (row := 85) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_9
  simp only [← c3407, ← c3408, k3, ← c3409] at e_85_9
  have e_86_4 := arithEq_of_rows h (row := 86) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_4
  simp only [← c3401, ← c3402, ← c3403] at e_86_4
  have e_86_5 := arithEq_of_rows h (row := 86) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_5
  simp only [← c3404, ← c3405, ← c3406] at e_86_5
  refine ⟨?_, ?_⟩
  · have hc := e_86_4
    simp only [e_85_8] at hc
    linear_combination c3410.trans k2 - hc
  · have hc := e_85_9
    simp only [e_85_7, e_86_5, e_85_8] at hc
    linear_combination c3411.trans k2 - hc

theorem privateBatchWrapper8_f389 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 38)) (a (.virt 481)) (a (.virt 482)) := by
  have c3412 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3413 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3414 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3415 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3416 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3417 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3418 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3419 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3420 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3421 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3422 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3423 := (privateBatchWrapper8_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3424 := (privateBatchWrapper8_copies107 a h).1
  have c3425 := (privateBatchWrapper8_copies107 a h).2.1
  have c3426 := (privateBatchWrapper8_copies107 a h).2.2.1
  have c3427 := (privateBatchWrapper8_copies107 a h).2.2.2.1
  have c3428 := (privateBatchWrapper8_copies107 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_85_10 := arithEq_of_rows h (row := 85) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_10
  simp only [← c3412, k3, ← c3413, k3, ← c3414] at e_85_10
  have e_85_11 := arithEq_of_rows h (row := 85) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_11
  simp only [← c3415, ← c3416, k3, ← c3417] at e_85_11
  have e_85_12 := arithEq_of_rows h (row := 85) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_12
  simp only [← c3424, ← c3425, k3, ← c3426] at e_85_12
  have e_86_6 := arithEq_of_rows h (row := 86) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_6
  simp only [← c3418, ← c3419, ← c3420] at e_86_6
  have e_86_7 := arithEq_of_rows h (row := 86) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_7
  simp only [← c3421, ← c3422, ← c3423] at e_86_7
  refine ⟨?_, ?_⟩
  · have hc := e_86_6
    simp only [e_85_11] at hc
    linear_combination c3427.trans k2 - hc
  · have hc := e_85_12
    simp only [e_85_10, e_86_7, e_85_11] at hc
    linear_combination c3428.trans k2 - hc

theorem privateBatchWrapper8_f390 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 39)) (a (.virt 483)) (a (.virt 484)) := by
  have c3429 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.1
  have c3430 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.1
  have c3431 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.1
  have c3432 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.1
  have c3433 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.1
  have c3434 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3435 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3436 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3437 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3438 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3439 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3440 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3441 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3442 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3443 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3444 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3445 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_85_13 := arithEq_of_rows h (row := 85) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_13
  simp only [← c3429, k3, ← c3430, k3, ← c3431] at e_85_13
  have e_85_14 := arithEq_of_rows h (row := 85) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_14
  simp only [← c3432, ← c3433, k3, ← c3434] at e_85_14
  have e_86_8 := arithEq_of_rows h (row := 86) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_8
  simp only [← c3435, ← c3436, ← c3437] at e_86_8
  have e_86_9 := arithEq_of_rows h (row := 86) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_9
  simp only [← c3438, ← c3439, ← c3440] at e_86_9
  have e_87_0 := arithEq_of_rows h (row := 87) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_0
  simp only [← c3441, ← c3442, k3, ← c3443] at e_87_0
  refine ⟨?_, ?_⟩
  · have hc := e_86_8
    simp only [e_85_14] at hc
    linear_combination c3444.trans k2 - hc
  · have hc := e_87_0
    simp only [e_85_13, e_86_9, e_85_14] at hc
    linear_combination c3445.trans k2 - hc

theorem privateBatchWrapper8_f391 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 43) = band (a (.virt 477)) (a (.virt 479)) := by
  have c3446 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3447 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3448 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_10 := arithEq_of_rows h (row := 86) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_10
  simp only [← c3446, ← c3447, ← c3448] at e_86_10
  have hr := e_86_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f392 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 47) = band (a (.virt 481)) (a (.virt 483)) := by
  have c3449 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3450 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3451 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_11 := arithEq_of_rows h (row := 86) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_11
  simp only [← c3449, ← c3450, ← c3451] at e_86_11
  have hr := e_86_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f393 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 51) = band (a (.wire 86 43)) (a (.wire 86 47)) := by
  have c3452 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3453 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3454 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_12 := arithEq_of_rows h (row := 86) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_12
  simp only [← c3452, ← c3453, ← c3454] at e_86_12
  have hr := e_86_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f394 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 55) = band (a (.wire 86 7)) (a (.wire 86 51)) := by
  have c3455 := (privateBatchWrapper8_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3456 := (privateBatchWrapper8_copies108 a h).1
  have c3457 := (privateBatchWrapper8_copies108 a h).2.1
  have e_86_13 := arithEq_of_rows h (row := 86) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_13
  simp only [← c3455, ← c3456, ← c3457] at e_86_13
  have hr := e_86_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f395 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 55) = a (.virt 11) := by
  have c3458 := (privateBatchWrapper8_copies108 a h).2.2.1
  exact c3458

theorem privateBatchWrapper8_f396 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 7) = bnot (a (.wire 8 27)) := by
  have c673 := (privateBatchWrapper8_copies21 a h).2.1
  have c674 := (privateBatchWrapper8_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper8_copies21 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c673, k3, ← c674, k3, ← c675] at e_17_1
  have hr := e_17_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f397 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 86 59) = band (a (.wire 12 43)) (a (.wire 17 7)) := by
  have c3459 := (privateBatchWrapper8_copies108 a h).2.2.2.1
  have c3460 := (privateBatchWrapper8_copies108 a h).2.2.2.2.1
  have c3461 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.1
  have e_86_14 := arithEq_of_rows h (row := 86) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_14
  simp only [← c3459, ← c3460, ← c3461] at e_86_14
  have hr := e_86_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f398 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 14)) (a (.virt 47)) (a (.virt 485)) (a (.virt 486)) := by
  have c3462 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.1
  have c3463 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.1
  have c3464 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.1
  have c3465 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.1
  have c3466 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3467 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3468 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3469 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3470 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3471 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3472 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3473 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3474 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_87_1 := arithEq_of_rows h (row := 87) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_1
  simp only [← c3462, k3, ← c3463, k3, ← c3464] at e_87_1
  have e_87_2 := arithEq_of_rows h (row := 87) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_2
  simp only [← c3465, ← c3466, k3, ← c3467] at e_87_2
  have e_87_3 := arithEq_of_rows h (row := 87) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_3
  simp only [← c3474, ← c3475, k3, ← c3476] at e_87_3
  have e_88_0 := arithEq_of_rows h (row := 88) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_0
  simp only [← c3468, ← c3469, ← c3470] at e_88_0
  have e_88_1 := arithEq_of_rows h (row := 88) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_1
  simp only [← c3471, ← c3472, ← c3473] at e_88_1
  refine ⟨?_, ?_⟩
  · have hc := e_88_0
    simp only [e_87_2] at hc
    linear_combination c3477.trans k2 - hc
  · have hc := e_87_3
    simp only [e_87_1, e_88_1, e_87_2] at hc
    linear_combination c3478.trans k2 - hc

theorem privateBatchWrapper8_f399 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 48)) (a (.virt 487)) (a (.virt 488)) := by
  have c3479 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3480 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3481 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3482 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3483 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3484 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3485 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3486 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3487 := (privateBatchWrapper8_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3488 := (privateBatchWrapper8_copies109 a h).1
  have c3489 := (privateBatchWrapper8_copies109 a h).2.1
  have c3490 := (privateBatchWrapper8_copies109 a h).2.2.1
  have c3491 := (privateBatchWrapper8_copies109 a h).2.2.2.1
  have c3492 := (privateBatchWrapper8_copies109 a h).2.2.2.2.1
  have c3493 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.1
  have c3494 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.1
  have c3495 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_87_4 := arithEq_of_rows h (row := 87) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_4
  simp only [← c3479, k3, ← c3480, k3, ← c3481] at e_87_4
  have e_87_5 := arithEq_of_rows h (row := 87) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_5
  simp only [← c3482, ← c3483, k3, ← c3484] at e_87_5
  have e_87_6 := arithEq_of_rows h (row := 87) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_6
  simp only [← c3491, ← c3492, k3, ← c3493] at e_87_6
  have e_88_2 := arithEq_of_rows h (row := 88) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_2
  simp only [← c3485, ← c3486, ← c3487] at e_88_2
  have e_88_3 := arithEq_of_rows h (row := 88) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_3
  simp only [← c3488, ← c3489, ← c3490] at e_88_3
  refine ⟨?_, ?_⟩
  · have hc := e_88_2
    simp only [e_87_5] at hc
    linear_combination c3494.trans k2 - hc
  · have hc := e_87_6
    simp only [e_87_4, e_88_3, e_87_5] at hc
    linear_combination c3495.trans k2 - hc

theorem privateBatchWrapper8_f400 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 49)) (a (.virt 489)) (a (.virt 490)) := by
  have c3496 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.1
  have c3497 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.1
  have c3498 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3499 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3500 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3501 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3502 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3503 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3504 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3505 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3506 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3507 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3508 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3509 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3510 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3511 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3512 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_87_7 := arithEq_of_rows h (row := 87) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_7
  simp only [← c3496, k3, ← c3497, k3, ← c3498] at e_87_7
  have e_87_8 := arithEq_of_rows h (row := 87) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_8
  simp only [← c3499, ← c3500, k3, ← c3501] at e_87_8
  have e_87_9 := arithEq_of_rows h (row := 87) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_9
  simp only [← c3508, ← c3509, k3, ← c3510] at e_87_9
  have e_88_4 := arithEq_of_rows h (row := 88) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_4
  simp only [← c3502, ← c3503, ← c3504] at e_88_4
  have e_88_5 := arithEq_of_rows h (row := 88) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_5
  simp only [← c3505, ← c3506, ← c3507] at e_88_5
  refine ⟨?_, ?_⟩
  · have hc := e_88_4
    simp only [e_87_8] at hc
    linear_combination c3511.trans k2 - hc
  · have hc := e_87_9
    simp only [e_87_7, e_88_5, e_87_8] at hc
    linear_combination c3512.trans k2 - hc

theorem privateBatchWrapper8_f401 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 50)) (a (.virt 491)) (a (.virt 492)) := by
  have c3513 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3514 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3515 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3516 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3517 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3518 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3519 := (privateBatchWrapper8_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3520 := (privateBatchWrapper8_copies110 a h).1
  have c3521 := (privateBatchWrapper8_copies110 a h).2.1
  have c3522 := (privateBatchWrapper8_copies110 a h).2.2.1
  have c3523 := (privateBatchWrapper8_copies110 a h).2.2.2.1
  have c3524 := (privateBatchWrapper8_copies110 a h).2.2.2.2.1
  have c3525 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.1
  have c3526 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.1
  have c3527 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.1
  have c3528 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.1
  have c3529 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_87_10 := arithEq_of_rows h (row := 87) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_10
  simp only [← c3513, k3, ← c3514, k3, ← c3515] at e_87_10
  have e_87_11 := arithEq_of_rows h (row := 87) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_11
  simp only [← c3516, ← c3517, k3, ← c3518] at e_87_11
  have e_87_12 := arithEq_of_rows h (row := 87) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_12
  simp only [← c3525, ← c3526, k3, ← c3527] at e_87_12
  have e_88_6 := arithEq_of_rows h (row := 88) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_6
  simp only [← c3519, ← c3520, ← c3521] at e_88_6
  have e_88_7 := arithEq_of_rows h (row := 88) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_7
  simp only [← c3522, ← c3523, ← c3524] at e_88_7
  refine ⟨?_, ?_⟩
  · have hc := e_88_6
    simp only [e_87_11] at hc
    linear_combination c3528.trans k2 - hc
  · have hc := e_87_12
    simp only [e_87_10, e_88_7, e_87_11] at hc
    linear_combination c3529.trans k2 - hc

theorem privateBatchWrapper8_f402 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 88 35) = band (a (.virt 485)) (a (.virt 487)) := by
  have c3530 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3531 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3532 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_8 := arithEq_of_rows h (row := 88) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_8
  simp only [← c3530, ← c3531, ← c3532] at e_88_8
  have hr := e_88_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f403 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 88 39) = band (a (.virt 489)) (a (.virt 491)) := by
  have c3533 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3534 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3535 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_9 := arithEq_of_rows h (row := 88) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_9
  simp only [← c3533, ← c3534, ← c3535] at e_88_9
  have hr := e_88_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f404 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 88 43) = band (a (.wire 88 35)) (a (.wire 88 39)) := by
  have c3536 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3537 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3538 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_10 := arithEq_of_rows h (row := 88) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_10
  simp only [← c3536, ← c3537, ← c3538] at e_88_10
  have hr := e_88_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f405 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 88 47) = band (a (.wire 86 59)) (a (.wire 88 43)) := by
  have c3539 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3540 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3541 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_11 := arithEq_of_rows h (row := 88) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_11
  simp only [← c3539, ← c3540, ← c3541] at e_88_11
  have hr := e_88_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f406 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 88 47) = a (.virt 11) := by
  have c3542 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3542

theorem privateBatchWrapper8_f407 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f408 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 88 51) = band (a (.wire 12 43)) (a (.wire 17 55)) := by
  have c3543 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3544 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3545 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_12 := arithEq_of_rows h (row := 88) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_12
  simp only [← c3543, ← c3544, ← c3545] at e_88_12
  have hr := e_88_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f409 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 14)) (a (.virt 58)) (a (.virt 493)) (a (.virt 494)) := by
  have c3546 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3547 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3548 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3549 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3550 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3551 := (privateBatchWrapper8_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3552 := (privateBatchWrapper8_copies111 a h).1
  have c3553 := (privateBatchWrapper8_copies111 a h).2.1
  have c3554 := (privateBatchWrapper8_copies111 a h).2.2.1
  have c3555 := (privateBatchWrapper8_copies111 a h).2.2.2.1
  have c3556 := (privateBatchWrapper8_copies111 a h).2.2.2.2.1
  have c3557 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.1
  have c3558 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.1
  have c3559 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.1
  have c3560 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.1
  have c3561 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.1
  have c3562 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_87_13 := arithEq_of_rows h (row := 87) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_13
  simp only [← c3546, k3, ← c3547, k3, ← c3548] at e_87_13
  have e_87_14 := arithEq_of_rows h (row := 87) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_14
  simp only [← c3549, ← c3550, k3, ← c3551] at e_87_14
  have e_88_13 := arithEq_of_rows h (row := 88) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_13
  simp only [← c3552, ← c3553, ← c3554] at e_88_13
  have e_88_14 := arithEq_of_rows h (row := 88) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_14
  simp only [← c3555, ← c3556, ← c3557] at e_88_14
  have e_89_0 := arithEq_of_rows h (row := 89) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_0
  simp only [← c3558, ← c3559, k3, ← c3560] at e_89_0
  refine ⟨?_, ?_⟩
  · have hc := e_88_13
    simp only [e_87_14] at hc
    linear_combination c3561.trans k2 - hc
  · have hc := e_89_0
    simp only [e_87_13, e_88_14, e_87_14] at hc
    linear_combination c3562.trans k2 - hc

theorem privateBatchWrapper8_f410 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 59)) (a (.virt 495)) (a (.virt 496)) := by
  have c3563 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3564 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3565 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3566 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3567 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3568 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3569 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3570 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3571 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3572 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3573 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3574 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3575 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3576 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3577 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3578 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3579 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_89_1 := arithEq_of_rows h (row := 89) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_1
  simp only [← c3563, k3, ← c3564, k3, ← c3565] at e_89_1
  have e_89_2 := arithEq_of_rows h (row := 89) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_2
  simp only [← c3566, ← c3567, k3, ← c3568] at e_89_2
  have e_89_3 := arithEq_of_rows h (row := 89) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_3
  simp only [← c3575, ← c3576, k3, ← c3577] at e_89_3
  have e_90_0 := arithEq_of_rows h (row := 90) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_0
  simp only [← c3569, ← c3570, ← c3571] at e_90_0
  have e_90_1 := arithEq_of_rows h (row := 90) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_1
  simp only [← c3572, ← c3573, ← c3574] at e_90_1
  refine ⟨?_, ?_⟩
  · have hc := e_90_0
    simp only [e_89_2] at hc
    linear_combination c3578.trans k2 - hc
  · have hc := e_89_3
    simp only [e_89_1, e_90_1, e_89_2] at hc
    linear_combination c3579.trans k2 - hc

theorem privateBatchWrapper8_f411 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 60)) (a (.virt 497)) (a (.virt 498)) := by
  have c3580 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3581 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3582 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3583 := (privateBatchWrapper8_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3584 := (privateBatchWrapper8_copies112 a h).1
  have c3585 := (privateBatchWrapper8_copies112 a h).2.1
  have c3586 := (privateBatchWrapper8_copies112 a h).2.2.1
  have c3587 := (privateBatchWrapper8_copies112 a h).2.2.2.1
  have c3588 := (privateBatchWrapper8_copies112 a h).2.2.2.2.1
  have c3589 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.1
  have c3590 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.1
  have c3591 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.1
  have c3592 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.1
  have c3593 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.1
  have c3594 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3595 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3596 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_89_4 := arithEq_of_rows h (row := 89) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_4
  simp only [← c3580, k3, ← c3581, k3, ← c3582] at e_89_4
  have e_89_5 := arithEq_of_rows h (row := 89) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_5
  simp only [← c3583, ← c3584, k3, ← c3585] at e_89_5
  have e_89_6 := arithEq_of_rows h (row := 89) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_6
  simp only [← c3592, ← c3593, k3, ← c3594] at e_89_6
  have e_90_2 := arithEq_of_rows h (row := 90) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_2
  simp only [← c3586, ← c3587, ← c3588] at e_90_2
  have e_90_3 := arithEq_of_rows h (row := 90) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_3
  simp only [← c3589, ← c3590, ← c3591] at e_90_3
  refine ⟨?_, ?_⟩
  · have hc := e_90_2
    simp only [e_89_5] at hc
    linear_combination c3595.trans k2 - hc
  · have hc := e_89_6
    simp only [e_89_4, e_90_3, e_89_5] at hc
    linear_combination c3596.trans k2 - hc

theorem privateBatchWrapper8_f412 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 61)) (a (.virt 499)) (a (.virt 500)) := by
  have c3597 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3598 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3599 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3600 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3601 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3602 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3603 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3604 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3605 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3606 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3607 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3608 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3609 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3610 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3611 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3612 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3613 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_89_7 := arithEq_of_rows h (row := 89) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_7
  simp only [← c3597, k3, ← c3598, k3, ← c3599] at e_89_7
  have e_89_8 := arithEq_of_rows h (row := 89) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_8
  simp only [← c3600, ← c3601, k3, ← c3602] at e_89_8
  have e_89_9 := arithEq_of_rows h (row := 89) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_9
  simp only [← c3609, ← c3610, k3, ← c3611] at e_89_9
  have e_90_4 := arithEq_of_rows h (row := 90) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_4
  simp only [← c3603, ← c3604, ← c3605] at e_90_4
  have e_90_5 := arithEq_of_rows h (row := 90) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_5
  simp only [← c3606, ← c3607, ← c3608] at e_90_5
  refine ⟨?_, ?_⟩
  · have hc := e_90_4
    simp only [e_89_8] at hc
    linear_combination c3612.trans k2 - hc
  · have hc := e_89_9
    simp only [e_89_7, e_90_5, e_89_8] at hc
    linear_combination c3613.trans k2 - hc

theorem privateBatchWrapper8_f413 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 90 27) = band (a (.virt 493)) (a (.virt 495)) := by
  have c3614 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3615 := (privateBatchWrapper8_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3616 := (privateBatchWrapper8_copies113 a h).1
  have e_90_6 := arithEq_of_rows h (row := 90) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_6
  simp only [← c3614, ← c3615, ← c3616] at e_90_6
  have hr := e_90_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f414 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 90 31) = band (a (.virt 497)) (a (.virt 499)) := by
  have c3617 := (privateBatchWrapper8_copies113 a h).2.1
  have c3618 := (privateBatchWrapper8_copies113 a h).2.2.1
  have c3619 := (privateBatchWrapper8_copies113 a h).2.2.2.1
  have e_90_7 := arithEq_of_rows h (row := 90) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_7
  simp only [← c3617, ← c3618, ← c3619] at e_90_7
  have hr := e_90_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f415 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 90 35) = band (a (.wire 90 27)) (a (.wire 90 31)) := by
  have c3620 := (privateBatchWrapper8_copies113 a h).2.2.2.2.1
  have c3621 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.1
  have c3622 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.1
  have e_90_8 := arithEq_of_rows h (row := 90) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_8
  simp only [← c3620, ← c3621, ← c3622] at e_90_8
  have hr := e_90_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f416 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 90 39) = band (a (.wire 88 51)) (a (.wire 90 35)) := by
  have c3623 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.1
  have c3624 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.1
  have c3625 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.1
  have e_90_9 := arithEq_of_rows h (row := 90) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_9
  simp only [← c3623, ← c3624, ← c3625] at e_90_9
  have hr := e_90_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f417 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 90 39) = a (.virt 11) := by
  have c3626 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c3626

theorem privateBatchWrapper8_f418 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f419 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 90 43) = band (a (.wire 12 43)) (a (.wire 18 43)) := by
  have c3627 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3628 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3629 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_10 := arithEq_of_rows h (row := 90) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_10
  simp only [← c3627, ← c3628, ← c3629] at e_90_10
  have hr := e_90_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f420 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 14)) (a (.virt 69)) (a (.virt 501)) (a (.virt 502)) := by
  have c3630 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3631 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3632 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3633 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3634 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3635 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3636 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3637 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3638 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3639 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3640 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3641 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3642 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3643 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3644 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3645 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3646 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_89_10 := arithEq_of_rows h (row := 89) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_10
  simp only [← c3630, k3, ← c3631, k3, ← c3632] at e_89_10
  have e_89_11 := arithEq_of_rows h (row := 89) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_11
  simp only [← c3633, ← c3634, k3, ← c3635] at e_89_11
  have e_89_12 := arithEq_of_rows h (row := 89) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_12
  simp only [← c3642, ← c3643, k3, ← c3644] at e_89_12
  have e_90_11 := arithEq_of_rows h (row := 90) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_11
  simp only [← c3636, ← c3637, ← c3638] at e_90_11
  have e_90_12 := arithEq_of_rows h (row := 90) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_12
  simp only [← c3639, ← c3640, ← c3641] at e_90_12
  refine ⟨?_, ?_⟩
  · have hc := e_90_11
    simp only [e_89_11] at hc
    linear_combination c3645.trans k2 - hc
  · have hc := e_89_12
    simp only [e_89_10, e_90_12, e_89_11] at hc
    linear_combination c3646.trans k2 - hc

theorem privateBatchWrapper8_f421 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 70)) (a (.virt 503)) (a (.virt 504)) := by
  have c3647 := (privateBatchWrapper8_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3648 := (privateBatchWrapper8_copies114 a h).1
  have c3649 := (privateBatchWrapper8_copies114 a h).2.1
  have c3650 := (privateBatchWrapper8_copies114 a h).2.2.1
  have c3651 := (privateBatchWrapper8_copies114 a h).2.2.2.1
  have c3652 := (privateBatchWrapper8_copies114 a h).2.2.2.2.1
  have c3653 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.1
  have c3654 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.1
  have c3655 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.1
  have c3656 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.1
  have c3657 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.1
  have c3658 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3659 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3660 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3661 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3662 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3663 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_89_13 := arithEq_of_rows h (row := 89) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_13
  simp only [← c3647, k3, ← c3648, k3, ← c3649] at e_89_13
  have e_89_14 := arithEq_of_rows h (row := 89) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_14
  simp only [← c3650, ← c3651, k3, ← c3652] at e_89_14
  have e_90_13 := arithEq_of_rows h (row := 90) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_13
  simp only [← c3653, ← c3654, ← c3655] at e_90_13
  have e_90_14 := arithEq_of_rows h (row := 90) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_14
  simp only [← c3656, ← c3657, ← c3658] at e_90_14
  have e_91_0 := arithEq_of_rows h (row := 91) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_0
  simp only [← c3659, ← c3660, k3, ← c3661] at e_91_0
  refine ⟨?_, ?_⟩
  · have hc := e_90_13
    simp only [e_89_14] at hc
    linear_combination c3662.trans k2 - hc
  · have hc := e_91_0
    simp only [e_89_13, e_90_14, e_89_14] at hc
    linear_combination c3663.trans k2 - hc

theorem privateBatchWrapper8_f422 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 71)) (a (.virt 505)) (a (.virt 506)) := by
  have c3664 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3665 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3666 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3667 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3668 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3669 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3670 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3671 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3672 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3673 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3674 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3675 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3676 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3677 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3678 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3679 := (privateBatchWrapper8_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3680 := (privateBatchWrapper8_copies115 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_91_1 := arithEq_of_rows h (row := 91) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_1
  simp only [← c3664, k3, ← c3665, k3, ← c3666] at e_91_1
  have e_91_2 := arithEq_of_rows h (row := 91) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_2
  simp only [← c3667, ← c3668, k3, ← c3669] at e_91_2
  have e_91_3 := arithEq_of_rows h (row := 91) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_3
  simp only [← c3676, ← c3677, k3, ← c3678] at e_91_3
  have e_92_0 := arithEq_of_rows h (row := 92) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_0
  simp only [← c3670, ← c3671, ← c3672] at e_92_0
  have e_92_1 := arithEq_of_rows h (row := 92) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_1
  simp only [← c3673, ← c3674, ← c3675] at e_92_1
  refine ⟨?_, ?_⟩
  · have hc := e_92_0
    simp only [e_91_2] at hc
    linear_combination c3679.trans k2 - hc
  · have hc := e_91_3
    simp only [e_91_1, e_92_1, e_91_2] at hc
    linear_combination c3680.trans k2 - hc

theorem privateBatchWrapper8_f423 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 72)) (a (.virt 507)) (a (.virt 508)) := by
  have c3681 := (privateBatchWrapper8_copies115 a h).2.1
  have c3682 := (privateBatchWrapper8_copies115 a h).2.2.1
  have c3683 := (privateBatchWrapper8_copies115 a h).2.2.2.1
  have c3684 := (privateBatchWrapper8_copies115 a h).2.2.2.2.1
  have c3685 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.1
  have c3686 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.1
  have c3687 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.1
  have c3688 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.1
  have c3689 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.1
  have c3690 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3691 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3692 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3693 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3694 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3695 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3696 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3697 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_91_4 := arithEq_of_rows h (row := 91) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_4
  simp only [← c3681, k3, ← c3682, k3, ← c3683] at e_91_4
  have e_91_5 := arithEq_of_rows h (row := 91) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_5
  simp only [← c3684, ← c3685, k3, ← c3686] at e_91_5
  have e_91_6 := arithEq_of_rows h (row := 91) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_6
  simp only [← c3693, ← c3694, k3, ← c3695] at e_91_6
  have e_92_2 := arithEq_of_rows h (row := 92) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_2
  simp only [← c3687, ← c3688, ← c3689] at e_92_2
  have e_92_3 := arithEq_of_rows h (row := 92) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_3
  simp only [← c3690, ← c3691, ← c3692] at e_92_3
  refine ⟨?_, ?_⟩
  · have hc := e_92_2
    simp only [e_91_5] at hc
    linear_combination c3696.trans k2 - hc
  · have hc := e_91_6
    simp only [e_91_4, e_92_3, e_91_5] at hc
    linear_combination c3697.trans k2 - hc

theorem privateBatchWrapper8_f424 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 92 19) = band (a (.virt 501)) (a (.virt 503)) := by
  have c3698 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3699 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3700 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_4 := arithEq_of_rows h (row := 92) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_4
  simp only [← c3698, ← c3699, ← c3700] at e_92_4
  have hr := e_92_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f425 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 92 23) = band (a (.virt 505)) (a (.virt 507)) := by
  have c3701 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3702 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3703 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_5 := arithEq_of_rows h (row := 92) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_5
  simp only [← c3701, ← c3702, ← c3703] at e_92_5
  have hr := e_92_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f426 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 92 27) = band (a (.wire 92 19)) (a (.wire 92 23)) := by
  have c3704 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3705 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3706 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_6 := arithEq_of_rows h (row := 92) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_6
  simp only [← c3704, ← c3705, ← c3706] at e_92_6
  have hr := e_92_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f427 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 92 31) = band (a (.wire 90 43)) (a (.wire 92 27)) := by
  have c3707 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3708 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3709 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_7 := arithEq_of_rows h (row := 92) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_7
  simp only [← c3707, ← c3708, ← c3709] at e_92_7
  have hr := e_92_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f428 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 92 31) = a (.virt 11) := by
  have c3710 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3710

theorem privateBatchWrapper8_f429 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f430 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 92 35) = band (a (.wire 12 43)) (a (.wire 20 31)) := by
  have c3711 := (privateBatchWrapper8_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3712 := (privateBatchWrapper8_copies116 a h).1
  have c3713 := (privateBatchWrapper8_copies116 a h).2.1
  have e_92_8 := arithEq_of_rows h (row := 92) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_8
  simp only [← c3711, ← c3712, ← c3713] at e_92_8
  have hr := e_92_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f431 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 14)) (a (.virt 80)) (a (.virt 509)) (a (.virt 510)) := by
  have c3714 := (privateBatchWrapper8_copies116 a h).2.2.1
  have c3715 := (privateBatchWrapper8_copies116 a h).2.2.2.1
  have c3716 := (privateBatchWrapper8_copies116 a h).2.2.2.2.1
  have c3717 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.1
  have c3718 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.1
  have c3719 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.1
  have c3720 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.1
  have c3721 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.1
  have c3722 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3723 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3724 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3725 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3726 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3727 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3728 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3729 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3730 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_91_7 := arithEq_of_rows h (row := 91) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_7
  simp only [← c3714, k3, ← c3715, k3, ← c3716] at e_91_7
  have e_91_8 := arithEq_of_rows h (row := 91) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_8
  simp only [← c3717, ← c3718, k3, ← c3719] at e_91_8
  have e_91_9 := arithEq_of_rows h (row := 91) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_9
  simp only [← c3726, ← c3727, k3, ← c3728] at e_91_9
  have e_92_9 := arithEq_of_rows h (row := 92) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_9
  simp only [← c3720, ← c3721, ← c3722] at e_92_9
  have e_92_10 := arithEq_of_rows h (row := 92) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_10
  simp only [← c3723, ← c3724, ← c3725] at e_92_10
  refine ⟨?_, ?_⟩
  · have hc := e_92_9
    simp only [e_91_8] at hc
    linear_combination c3729.trans k2 - hc
  · have hc := e_91_9
    simp only [e_91_7, e_92_10, e_91_8] at hc
    linear_combination c3730.trans k2 - hc

theorem privateBatchWrapper8_f432 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 81)) (a (.virt 511)) (a (.virt 512)) := by
  have c3731 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3732 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3733 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3734 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3735 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3736 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3737 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3738 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3739 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3740 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3741 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3742 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3743 := (privateBatchWrapper8_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3744 := (privateBatchWrapper8_copies117 a h).1
  have c3745 := (privateBatchWrapper8_copies117 a h).2.1
  have c3746 := (privateBatchWrapper8_copies117 a h).2.2.1
  have c3747 := (privateBatchWrapper8_copies117 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_91_10 := arithEq_of_rows h (row := 91) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_10
  simp only [← c3731, k3, ← c3732, k3, ← c3733] at e_91_10
  have e_91_11 := arithEq_of_rows h (row := 91) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_11
  simp only [← c3734, ← c3735, k3, ← c3736] at e_91_11
  have e_91_12 := arithEq_of_rows h (row := 91) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_12
  simp only [← c3743, ← c3744, k3, ← c3745] at e_91_12
  have e_92_11 := arithEq_of_rows h (row := 92) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_11
  simp only [← c3737, ← c3738, ← c3739] at e_92_11
  have e_92_12 := arithEq_of_rows h (row := 92) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_12
  simp only [← c3740, ← c3741, ← c3742] at e_92_12
  refine ⟨?_, ?_⟩
  · have hc := e_92_11
    simp only [e_91_11] at hc
    linear_combination c3746.trans k2 - hc
  · have hc := e_91_12
    simp only [e_91_10, e_92_12, e_91_11] at hc
    linear_combination c3747.trans k2 - hc

theorem privateBatchWrapper8_f433 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 82)) (a (.virt 513)) (a (.virt 514)) := by
  have c3748 := (privateBatchWrapper8_copies117 a h).2.2.2.2.1
  have c3749 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.1
  have c3750 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.1
  have c3751 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.1
  have c3752 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.1
  have c3753 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.1
  have c3754 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3755 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3756 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3757 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3758 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3759 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3760 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3761 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3762 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3763 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3764 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_91_13 := arithEq_of_rows h (row := 91) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_13
  simp only [← c3748, k3, ← c3749, k3, ← c3750] at e_91_13
  have e_91_14 := arithEq_of_rows h (row := 91) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_14
  simp only [← c3751, ← c3752, k3, ← c3753] at e_91_14
  have e_92_13 := arithEq_of_rows h (row := 92) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_13
  simp only [← c3754, ← c3755, ← c3756] at e_92_13
  have e_92_14 := arithEq_of_rows h (row := 92) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_14
  simp only [← c3757, ← c3758, ← c3759] at e_92_14
  have e_93_0 := arithEq_of_rows h (row := 93) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_0
  simp only [← c3760, ← c3761, k3, ← c3762] at e_93_0
  refine ⟨?_, ?_⟩
  · have hc := e_92_13
    simp only [e_91_14] at hc
    linear_combination c3763.trans k2 - hc
  · have hc := e_93_0
    simp only [e_91_13, e_92_14, e_91_14] at hc
    linear_combination c3764.trans k2 - hc

theorem privateBatchWrapper8_f434 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 83)) (a (.virt 515)) (a (.virt 516)) := by
  have c3765 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3766 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3767 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3768 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3769 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3770 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3771 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3772 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3773 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3774 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3775 := (privateBatchWrapper8_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3776 := (privateBatchWrapper8_copies118 a h).1
  have c3777 := (privateBatchWrapper8_copies118 a h).2.1
  have c3778 := (privateBatchWrapper8_copies118 a h).2.2.1
  have c3779 := (privateBatchWrapper8_copies118 a h).2.2.2.1
  have c3780 := (privateBatchWrapper8_copies118 a h).2.2.2.2.1
  have c3781 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_93_1 := arithEq_of_rows h (row := 93) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_1
  simp only [← c3765, k3, ← c3766, k3, ← c3767] at e_93_1
  have e_93_2 := arithEq_of_rows h (row := 93) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_2
  simp only [← c3768, ← c3769, k3, ← c3770] at e_93_2
  have e_93_3 := arithEq_of_rows h (row := 93) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_3
  simp only [← c3777, ← c3778, k3, ← c3779] at e_93_3
  have e_94_0 := arithEq_of_rows h (row := 94) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_0
  simp only [← c3771, ← c3772, ← c3773] at e_94_0
  have e_94_1 := arithEq_of_rows h (row := 94) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_1
  simp only [← c3774, ← c3775, ← c3776] at e_94_1
  refine ⟨?_, ?_⟩
  · have hc := e_94_0
    simp only [e_93_2] at hc
    linear_combination c3780.trans k2 - hc
  · have hc := e_93_3
    simp only [e_93_1, e_94_1, e_93_2] at hc
    linear_combination c3781.trans k2 - hc

theorem privateBatchWrapper8_f435 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 94 11) = band (a (.virt 509)) (a (.virt 511)) := by
  have c3782 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.1
  have c3783 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.1
  have c3784 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.1
  have e_94_2 := arithEq_of_rows h (row := 94) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_2
  simp only [← c3782, ← c3783, ← c3784] at e_94_2
  have hr := e_94_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f436 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 94 15) = band (a (.virt 513)) (a (.virt 515)) := by
  have c3785 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.1
  have c3786 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3787 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_3 := arithEq_of_rows h (row := 94) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_3
  simp only [← c3785, ← c3786, ← c3787] at e_94_3
  have hr := e_94_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f437 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 94 19) = band (a (.wire 94 11)) (a (.wire 94 15)) := by
  have c3788 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3789 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3790 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_4 := arithEq_of_rows h (row := 94) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_4
  simp only [← c3788, ← c3789, ← c3790] at e_94_4
  have hr := e_94_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f438 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 94 23) = band (a (.wire 92 35)) (a (.wire 94 19)) := by
  have c3791 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3792 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3793 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_5 := arithEq_of_rows h (row := 94) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_5
  simp only [← c3791, ← c3792, ← c3793] at e_94_5
  have hr := e_94_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f439 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 94 23) = a (.virt 11) := by
  have c3794 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3794

theorem privateBatchWrapper8_f440 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 13 31) = bnot (a (.wire 4 59)) := by
  have c583 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper8_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c583, k3, ← c584, k3, ← c585] at e_13_7
  have hr := e_13_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f441 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 19) = bnot (a (.wire 6 43)) := by
  have c628 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c628, k3, ← c629, k3, ← c630] at e_15_4
  have hr := e_15_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f442 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 94 27) = band (a (.wire 13 31)) (a (.wire 15 19)) := by
  have c3795 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3796 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3797 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_6 := arithEq_of_rows h (row := 94) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_6
  simp only [← c3795, ← c3796, ← c3797] at e_94_6
  have hr := e_94_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f443 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 25)) (a (.virt 36)) (a (.virt 517)) (a (.virt 518)) := by
  have c3798 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3799 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3800 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3801 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3802 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3803 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3804 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3805 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3806 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3807 := (privateBatchWrapper8_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3808 := (privateBatchWrapper8_copies119 a h).1
  have c3809 := (privateBatchWrapper8_copies119 a h).2.1
  have c3810 := (privateBatchWrapper8_copies119 a h).2.2.1
  have c3811 := (privateBatchWrapper8_copies119 a h).2.2.2.1
  have c3812 := (privateBatchWrapper8_copies119 a h).2.2.2.2.1
  have c3813 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.1
  have c3814 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_93_4 := arithEq_of_rows h (row := 93) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_4
  simp only [← c3798, k3, ← c3799, k3, ← c3800] at e_93_4
  have e_93_5 := arithEq_of_rows h (row := 93) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_5
  simp only [← c3801, ← c3802, k3, ← c3803] at e_93_5
  have e_93_6 := arithEq_of_rows h (row := 93) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_6
  simp only [← c3810, ← c3811, k3, ← c3812] at e_93_6
  have e_94_7 := arithEq_of_rows h (row := 94) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_7
  simp only [← c3804, ← c3805, ← c3806] at e_94_7
  have e_94_8 := arithEq_of_rows h (row := 94) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_8
  simp only [← c3807, ← c3808, ← c3809] at e_94_8
  refine ⟨?_, ?_⟩
  · have hc := e_94_7
    simp only [e_93_5] at hc
    linear_combination c3813.trans k2 - hc
  · have hc := e_93_6
    simp only [e_93_4, e_94_8, e_93_5] at hc
    linear_combination c3814.trans k2 - hc

theorem privateBatchWrapper8_f444 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 26)) (a (.virt 37)) (a (.virt 519)) (a (.virt 520)) := by
  have c3815 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.1
  have c3816 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.1
  have c3817 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.1
  have c3818 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3819 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3820 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3821 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3822 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3823 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3824 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3825 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3826 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3827 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3828 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3829 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3830 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3831 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_93_7 := arithEq_of_rows h (row := 93) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_7
  simp only [← c3815, k3, ← c3816, k3, ← c3817] at e_93_7
  have e_93_8 := arithEq_of_rows h (row := 93) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_8
  simp only [← c3818, ← c3819, k3, ← c3820] at e_93_8
  have e_93_9 := arithEq_of_rows h (row := 93) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_9
  simp only [← c3827, ← c3828, k3, ← c3829] at e_93_9
  have e_94_9 := arithEq_of_rows h (row := 94) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_9
  simp only [← c3821, ← c3822, ← c3823] at e_94_9
  have e_94_10 := arithEq_of_rows h (row := 94) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_10
  simp only [← c3824, ← c3825, ← c3826] at e_94_10
  refine ⟨?_, ?_⟩
  · have hc := e_94_9
    simp only [e_93_8] at hc
    linear_combination c3830.trans k2 - hc
  · have hc := e_93_9
    simp only [e_93_7, e_94_10, e_93_8] at hc
    linear_combination c3831.trans k2 - hc

theorem privateBatchWrapper8_f445 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 27)) (a (.virt 38)) (a (.virt 521)) (a (.virt 522)) := by
  have c3832 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3833 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3834 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3835 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3836 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3837 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3838 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3839 := (privateBatchWrapper8_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3840 := (privateBatchWrapper8_copies120 a h).1
  have c3841 := (privateBatchWrapper8_copies120 a h).2.1
  have c3842 := (privateBatchWrapper8_copies120 a h).2.2.1
  have c3843 := (privateBatchWrapper8_copies120 a h).2.2.2.1
  have c3844 := (privateBatchWrapper8_copies120 a h).2.2.2.2.1
  have c3845 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.1
  have c3846 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.1
  have c3847 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.1
  have c3848 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_93_10 := arithEq_of_rows h (row := 93) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_10
  simp only [← c3832, k3, ← c3833, k3, ← c3834] at e_93_10
  have e_93_11 := arithEq_of_rows h (row := 93) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_11
  simp only [← c3835, ← c3836, k3, ← c3837] at e_93_11
  have e_93_12 := arithEq_of_rows h (row := 93) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_12
  simp only [← c3844, ← c3845, k3, ← c3846] at e_93_12
  have e_94_11 := arithEq_of_rows h (row := 94) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_11
  simp only [← c3838, ← c3839, ← c3840] at e_94_11
  have e_94_12 := arithEq_of_rows h (row := 94) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_12
  simp only [← c3841, ← c3842, ← c3843] at e_94_12
  refine ⟨?_, ?_⟩
  · have hc := e_94_11
    simp only [e_93_11] at hc
    linear_combination c3847.trans k2 - hc
  · have hc := e_93_12
    simp only [e_93_10, e_94_12, e_93_11] at hc
    linear_combination c3848.trans k2 - hc

theorem privateBatchWrapper8_f446 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 28)) (a (.virt 39)) (a (.virt 523)) (a (.virt 524)) := by
  have c3849 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.1
  have c3850 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3851 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3852 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3853 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3854 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3855 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3856 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3857 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3858 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3859 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3860 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3861 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3862 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3863 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3864 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3865 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_93_13 := arithEq_of_rows h (row := 93) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_13
  simp only [← c3849, k3, ← c3850, k3, ← c3851] at e_93_13
  have e_93_14 := arithEq_of_rows h (row := 93) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_14
  simp only [← c3852, ← c3853, k3, ← c3854] at e_93_14
  have e_94_13 := arithEq_of_rows h (row := 94) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_13
  simp only [← c3855, ← c3856, ← c3857] at e_94_13
  have e_94_14 := arithEq_of_rows h (row := 94) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_14
  simp only [← c3858, ← c3859, ← c3860] at e_94_14
  have e_95_0 := arithEq_of_rows h (row := 95) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_0
  simp only [← c3861, ← c3862, k3, ← c3863] at e_95_0
  refine ⟨?_, ?_⟩
  · have hc := e_94_13
    simp only [e_93_14] at hc
    linear_combination c3864.trans k2 - hc
  · have hc := e_95_0
    simp only [e_93_13, e_94_14, e_93_14] at hc
    linear_combination c3865.trans k2 - hc

theorem privateBatchWrapper8_f447 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 3) = band (a (.virt 517)) (a (.virt 519)) := by
  have c3866 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3867 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3868 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_0 := arithEq_of_rows h (row := 96) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_0
  simp only [← c3866, ← c3867, ← c3868] at e_96_0
  have hr := e_96_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f448 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 7) = band (a (.virt 521)) (a (.virt 523)) := by
  have c3869 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3870 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3871 := (privateBatchWrapper8_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_96_1 := arithEq_of_rows h (row := 96) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_1
  simp only [← c3869, ← c3870, ← c3871] at e_96_1
  have hr := e_96_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f449 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 11) = band (a (.wire 96 3)) (a (.wire 96 7)) := by
  have c3872 := (privateBatchWrapper8_copies121 a h).1
  have c3873 := (privateBatchWrapper8_copies121 a h).2.1
  have c3874 := (privateBatchWrapper8_copies121 a h).2.2.1
  have e_96_2 := arithEq_of_rows h (row := 96) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_2
  simp only [← c3872, ← c3873, ← c3874] at e_96_2
  have hr := e_96_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f450 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 15) = band (a (.wire 94 27)) (a (.wire 96 11)) := by
  have c3875 := (privateBatchWrapper8_copies121 a h).2.2.2.1
  have c3876 := (privateBatchWrapper8_copies121 a h).2.2.2.2.1
  have c3877 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.1
  have e_96_3 := arithEq_of_rows h (row := 96) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_3
  simp only [← c3875, ← c3876, ← c3877] at e_96_3
  have hr := e_96_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f451 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 15) = a (.virt 11) := by
  have c3878 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.1
  exact c3878

theorem privateBatchWrapper8_f452 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 7) = bnot (a (.wire 8 27)) := by
  have c673 := (privateBatchWrapper8_copies21 a h).2.1
  have c674 := (privateBatchWrapper8_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper8_copies21 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c673, k3, ← c674, k3, ← c675] at e_17_1
  have hr := e_17_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f453 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 19) = band (a (.wire 13 31)) (a (.wire 17 7)) := by
  have c3879 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.1
  have c3880 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.1
  have c3881 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.1
  have e_96_4 := arithEq_of_rows h (row := 96) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_4
  simp only [← c3879, ← c3880, ← c3881] at e_96_4
  have hr := e_96_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f454 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 25)) (a (.virt 47)) (a (.virt 525)) (a (.virt 526)) := by
  have c3882 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3883 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3884 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3885 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3886 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3887 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3888 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3889 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3890 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3891 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3892 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3893 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3894 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3895 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3896 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3897 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3898 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_95_1 := arithEq_of_rows h (row := 95) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_1
  simp only [← c3882, k3, ← c3883, k3, ← c3884] at e_95_1
  have e_95_2 := arithEq_of_rows h (row := 95) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_2
  simp only [← c3885, ← c3886, k3, ← c3887] at e_95_2
  have e_95_3 := arithEq_of_rows h (row := 95) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_3
  simp only [← c3894, ← c3895, k3, ← c3896] at e_95_3
  have e_96_5 := arithEq_of_rows h (row := 96) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_5
  simp only [← c3888, ← c3889, ← c3890] at e_96_5
  have e_96_6 := arithEq_of_rows h (row := 96) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_6
  simp only [← c3891, ← c3892, ← c3893] at e_96_6
  refine ⟨?_, ?_⟩
  · have hc := e_96_5
    simp only [e_95_2] at hc
    linear_combination c3897.trans k2 - hc
  · have hc := e_95_3
    simp only [e_95_1, e_96_6, e_95_2] at hc
    linear_combination c3898.trans k2 - hc

theorem privateBatchWrapper8_f455 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 26)) (a (.virt 48)) (a (.virt 527)) (a (.virt 528)) := by
  have c3899 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3900 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3901 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3902 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3903 := (privateBatchWrapper8_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3904 := (privateBatchWrapper8_copies122 a h).1
  have c3905 := (privateBatchWrapper8_copies122 a h).2.1
  have c3906 := (privateBatchWrapper8_copies122 a h).2.2.1
  have c3907 := (privateBatchWrapper8_copies122 a h).2.2.2.1
  have c3908 := (privateBatchWrapper8_copies122 a h).2.2.2.2.1
  have c3909 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.1
  have c3910 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.1
  have c3911 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.1
  have c3912 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.1
  have c3913 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.1
  have c3914 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3915 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_95_4 := arithEq_of_rows h (row := 95) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_4
  simp only [← c3899, k3, ← c3900, k3, ← c3901] at e_95_4
  have e_95_5 := arithEq_of_rows h (row := 95) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_5
  simp only [← c3902, ← c3903, k3, ← c3904] at e_95_5
  have e_95_6 := arithEq_of_rows h (row := 95) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_6
  simp only [← c3911, ← c3912, k3, ← c3913] at e_95_6
  have e_96_7 := arithEq_of_rows h (row := 96) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_7
  simp only [← c3905, ← c3906, ← c3907] at e_96_7
  have e_96_8 := arithEq_of_rows h (row := 96) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_8
  simp only [← c3908, ← c3909, ← c3910] at e_96_8
  refine ⟨?_, ?_⟩
  · have hc := e_96_7
    simp only [e_95_5] at hc
    linear_combination c3914.trans k2 - hc
  · have hc := e_95_6
    simp only [e_95_4, e_96_8, e_95_5] at hc
    linear_combination c3915.trans k2 - hc

theorem privateBatchWrapper8_f456 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 27)) (a (.virt 49)) (a (.virt 529)) (a (.virt 530)) := by
  have c3916 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3917 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3918 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3919 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3920 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3921 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3922 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3923 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3924 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3925 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3926 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3927 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3928 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3929 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3930 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3931 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3932 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_95_7 := arithEq_of_rows h (row := 95) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_7
  simp only [← c3916, k3, ← c3917, k3, ← c3918] at e_95_7
  have e_95_8 := arithEq_of_rows h (row := 95) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_8
  simp only [← c3919, ← c3920, k3, ← c3921] at e_95_8
  have e_95_9 := arithEq_of_rows h (row := 95) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_9
  simp only [← c3928, ← c3929, k3, ← c3930] at e_95_9
  have e_96_9 := arithEq_of_rows h (row := 96) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_9
  simp only [← c3922, ← c3923, ← c3924] at e_96_9
  have e_96_10 := arithEq_of_rows h (row := 96) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_10
  simp only [← c3925, ← c3926, ← c3927] at e_96_10
  refine ⟨?_, ?_⟩
  · have hc := e_96_9
    simp only [e_95_8] at hc
    linear_combination c3931.trans k2 - hc
  · have hc := e_95_9
    simp only [e_95_7, e_96_10, e_95_8] at hc
    linear_combination c3932.trans k2 - hc

theorem privateBatchWrapper8_f457 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 28)) (a (.virt 50)) (a (.virt 531)) (a (.virt 532)) := by
  have c3933 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3934 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3935 := (privateBatchWrapper8_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3936 := (privateBatchWrapper8_copies123 a h).1
  have c3937 := (privateBatchWrapper8_copies123 a h).2.1
  have c3938 := (privateBatchWrapper8_copies123 a h).2.2.1
  have c3939 := (privateBatchWrapper8_copies123 a h).2.2.2.1
  have c3940 := (privateBatchWrapper8_copies123 a h).2.2.2.2.1
  have c3941 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.1
  have c3942 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.1
  have c3943 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.1
  have c3944 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.1
  have c3945 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.1
  have c3946 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3947 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3948 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3949 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_95_10 := arithEq_of_rows h (row := 95) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_10
  simp only [← c3933, k3, ← c3934, k3, ← c3935] at e_95_10
  have e_95_11 := arithEq_of_rows h (row := 95) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_11
  simp only [← c3936, ← c3937, k3, ← c3938] at e_95_11
  have e_95_12 := arithEq_of_rows h (row := 95) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_12
  simp only [← c3945, ← c3946, k3, ← c3947] at e_95_12
  have e_96_11 := arithEq_of_rows h (row := 96) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_11
  simp only [← c3939, ← c3940, ← c3941] at e_96_11
  have e_96_12 := arithEq_of_rows h (row := 96) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_12
  simp only [← c3942, ← c3943, ← c3944] at e_96_12
  refine ⟨?_, ?_⟩
  · have hc := e_96_11
    simp only [e_95_11] at hc
    linear_combination c3948.trans k2 - hc
  · have hc := e_95_12
    simp only [e_95_10, e_96_12, e_95_11] at hc
    linear_combination c3949.trans k2 - hc

theorem privateBatchWrapper8_f458 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 55) = band (a (.virt 525)) (a (.virt 527)) := by
  have c3950 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3951 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3952 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_13 := arithEq_of_rows h (row := 96) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_13
  simp only [← c3950, ← c3951, ← c3952] at e_96_13
  have hr := e_96_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f459 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 96 59) = band (a (.virt 529)) (a (.virt 531)) := by
  have c3953 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_14 := arithEq_of_rows h (row := 96) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_14
  simp only [← c3953, ← c3954, ← c3955] at e_96_14
  have hr := e_96_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f460 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 3) = band (a (.wire 96 55)) (a (.wire 96 59)) := by
  have c3956 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3958 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_0 := arithEq_of_rows h (row := 97) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_0
  simp only [← c3956, ← c3957, ← c3958] at e_97_0
  have hr := e_97_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f461 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 7) = band (a (.wire 96 19)) (a (.wire 97 3)) := by
  have c3959 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3960 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3961 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_1 := arithEq_of_rows h (row := 97) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_1
  simp only [← c3959, ← c3960, ← c3961] at e_97_1
  have hr := e_97_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f462 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 7) = a (.virt 11) := by
  have c3962 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c3962

theorem privateBatchWrapper8_f463 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f464 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 11) = band (a (.wire 13 31)) (a (.wire 17 55)) := by
  have c3963 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3964 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3965 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_2 := arithEq_of_rows h (row := 97) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_2
  simp only [← c3963, ← c3964, ← c3965] at e_97_2
  have hr := e_97_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f465 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 25)) (a (.virt 58)) (a (.virt 533)) (a (.virt 534)) := by
  have c3966 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3967 := (privateBatchWrapper8_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3968 := (privateBatchWrapper8_copies124 a h).1
  have c3969 := (privateBatchWrapper8_copies124 a h).2.1
  have c3970 := (privateBatchWrapper8_copies124 a h).2.2.1
  have c3971 := (privateBatchWrapper8_copies124 a h).2.2.2.1
  have c3972 := (privateBatchWrapper8_copies124 a h).2.2.2.2.1
  have c3973 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.1
  have c3974 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.1
  have c3975 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.1
  have c3976 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.1
  have c3977 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.1
  have c3978 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3979 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3980 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3981 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3982 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_95_13 := arithEq_of_rows h (row := 95) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_13
  simp only [← c3966, k3, ← c3967, k3, ← c3968] at e_95_13
  have e_95_14 := arithEq_of_rows h (row := 95) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_14
  simp only [← c3969, ← c3970, k3, ← c3971] at e_95_14
  have e_97_3 := arithEq_of_rows h (row := 97) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_3
  simp only [← c3972, ← c3973, ← c3974] at e_97_3
  have e_97_4 := arithEq_of_rows h (row := 97) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_4
  simp only [← c3975, ← c3976, ← c3977] at e_97_4
  have e_98_0 := arithEq_of_rows h (row := 98) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_0
  simp only [← c3978, ← c3979, k3, ← c3980] at e_98_0
  refine ⟨?_, ?_⟩
  · have hc := e_97_3
    simp only [e_95_14] at hc
    linear_combination c3981.trans k2 - hc
  · have hc := e_98_0
    simp only [e_95_13, e_97_4, e_95_14] at hc
    linear_combination c3982.trans k2 - hc

theorem privateBatchWrapper8_f466 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 26)) (a (.virt 59)) (a (.virt 535)) (a (.virt 536)) := by
  have c3983 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3984 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3985 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3986 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3987 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3988 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3989 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3990 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3991 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3992 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3993 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3994 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3995 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3996 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3997 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3998 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3999 := (privateBatchWrapper8_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_97_5 := arithEq_of_rows h (row := 97) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_5
  simp only [← c3989, ← c3990, ← c3991] at e_97_5
  have e_97_6 := arithEq_of_rows h (row := 97) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_6
  simp only [← c3992, ← c3993, ← c3994] at e_97_6
  have e_98_1 := arithEq_of_rows h (row := 98) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_1
  simp only [← c3983, k3, ← c3984, k3, ← c3985] at e_98_1
  have e_98_2 := arithEq_of_rows h (row := 98) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c3986, ← c3987, k3, ← c3988] at e_98_2
  have e_98_3 := arithEq_of_rows h (row := 98) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_3
  simp only [← c3995, ← c3996, k3, ← c3997] at e_98_3
  refine ⟨?_, ?_⟩
  · have hc := e_97_5
    simp only [e_98_2] at hc
    linear_combination c3998.trans k2 - hc
  · have hc := e_98_3
    simp only [e_98_1, e_97_6, e_98_2] at hc
    linear_combination c3999.trans k2 - hc

theorem privateBatchWrapper8_f467 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 27)) (a (.virt 60)) (a (.virt 537)) (a (.virt 538)) := by
  have c4000 := (privateBatchWrapper8_copies125 a h).1
  have c4001 := (privateBatchWrapper8_copies125 a h).2.1
  have c4002 := (privateBatchWrapper8_copies125 a h).2.2.1
  have c4003 := (privateBatchWrapper8_copies125 a h).2.2.2.1
  have c4004 := (privateBatchWrapper8_copies125 a h).2.2.2.2.1
  have c4005 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.1
  have c4006 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.1
  have c4007 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.1
  have c4008 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.1
  have c4009 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.1
  have c4010 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4011 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4012 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4013 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4014 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4015 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4016 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_97_7 := arithEq_of_rows h (row := 97) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_7
  simp only [← c4006, ← c4007, ← c4008] at e_97_7
  have e_97_8 := arithEq_of_rows h (row := 97) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_8
  simp only [← c4009, ← c4010, ← c4011] at e_97_8
  have e_98_4 := arithEq_of_rows h (row := 98) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_4
  simp only [← c4000, k3, ← c4001, k3, ← c4002] at e_98_4
  have e_98_5 := arithEq_of_rows h (row := 98) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_5
  simp only [← c4003, ← c4004, k3, ← c4005] at e_98_5
  have e_98_6 := arithEq_of_rows h (row := 98) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_6
  simp only [← c4012, ← c4013, k3, ← c4014] at e_98_6
  refine ⟨?_, ?_⟩
  · have hc := e_97_7
    simp only [e_98_5] at hc
    linear_combination c4015.trans k2 - hc
  · have hc := e_98_6
    simp only [e_98_4, e_97_8, e_98_5] at hc
    linear_combination c4016.trans k2 - hc

theorem privateBatchWrapper8_f468 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 28)) (a (.virt 61)) (a (.virt 539)) (a (.virt 540)) := by
  have c4017 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4018 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4019 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4020 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4021 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4022 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4023 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4024 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4025 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4026 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4027 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4028 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4029 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4030 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4031 := (privateBatchWrapper8_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4032 := (privateBatchWrapper8_copies126 a h).1
  have c4033 := (privateBatchWrapper8_copies126 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_97_9 := arithEq_of_rows h (row := 97) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_9
  simp only [← c4023, ← c4024, ← c4025] at e_97_9
  have e_97_10 := arithEq_of_rows h (row := 97) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_10
  simp only [← c4026, ← c4027, ← c4028] at e_97_10
  have e_98_7 := arithEq_of_rows h (row := 98) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_7
  simp only [← c4017, k3, ← c4018, k3, ← c4019] at e_98_7
  have e_98_8 := arithEq_of_rows h (row := 98) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_8
  simp only [← c4020, ← c4021, k3, ← c4022] at e_98_8
  have e_98_9 := arithEq_of_rows h (row := 98) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_9
  simp only [← c4029, ← c4030, k3, ← c4031] at e_98_9
  refine ⟨?_, ?_⟩
  · have hc := e_97_9
    simp only [e_98_8] at hc
    linear_combination c4032.trans k2 - hc
  · have hc := e_98_9
    simp only [e_98_7, e_97_10, e_98_8] at hc
    linear_combination c4033.trans k2 - hc

theorem privateBatchWrapper8_f469 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 47) = band (a (.virt 533)) (a (.virt 535)) := by
  have c4034 := (privateBatchWrapper8_copies126 a h).2.2.1
  have c4035 := (privateBatchWrapper8_copies126 a h).2.2.2.1
  have c4036 := (privateBatchWrapper8_copies126 a h).2.2.2.2.1
  have e_97_11 := arithEq_of_rows h (row := 97) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_11
  simp only [← c4034, ← c4035, ← c4036] at e_97_11
  have hr := e_97_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f470 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 51) = band (a (.virt 537)) (a (.virt 539)) := by
  have c4037 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.1
  have c4038 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.1
  have c4039 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.1
  have e_97_12 := arithEq_of_rows h (row := 97) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_12
  simp only [← c4037, ← c4038, ← c4039] at e_97_12
  have hr := e_97_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f471 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 55) = band (a (.wire 97 47)) (a (.wire 97 51)) := by
  have c4040 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.1
  have c4041 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.1
  have c4042 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_97_13 := arithEq_of_rows h (row := 97) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_13
  simp only [← c4040, ← c4041, ← c4042] at e_97_13
  have hr := e_97_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f472 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 59) = band (a (.wire 97 11)) (a (.wire 97 55)) := by
  have c4043 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4044 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4045 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_14 := arithEq_of_rows h (row := 97) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_14
  simp only [← c4043, ← c4044, ← c4045] at e_97_14
  have hr := e_97_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f473 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 97 59) = a (.virt 11) := by
  have c4046 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4046

theorem privateBatchWrapper8_f474 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f475 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 3) = band (a (.wire 13 31)) (a (.wire 18 43)) := by
  have c4047 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4048 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4049 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_0 := arithEq_of_rows h (row := 99) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_0
  simp only [← c4047, ← c4048, ← c4049] at e_99_0
  have hr := e_99_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f476 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 25)) (a (.virt 69)) (a (.virt 541)) (a (.virt 542)) := by
  have c4050 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4051 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4052 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4053 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4054 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4055 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4056 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4057 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4058 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4059 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4060 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4061 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4062 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4063 := (privateBatchWrapper8_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4064 := (privateBatchWrapper8_copies127 a h).1
  have c4065 := (privateBatchWrapper8_copies127 a h).2.1
  have c4066 := (privateBatchWrapper8_copies127 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_98_10 := arithEq_of_rows h (row := 98) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_10
  simp only [← c4050, k3, ← c4051, k3, ← c4052] at e_98_10
  have e_98_11 := arithEq_of_rows h (row := 98) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_11
  simp only [← c4053, ← c4054, k3, ← c4055] at e_98_11
  have e_98_12 := arithEq_of_rows h (row := 98) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_12
  simp only [← c4062, ← c4063, k3, ← c4064] at e_98_12
  have e_99_1 := arithEq_of_rows h (row := 99) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_1
  simp only [← c4056, ← c4057, ← c4058] at e_99_1
  have e_99_2 := arithEq_of_rows h (row := 99) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_2
  simp only [← c4059, ← c4060, ← c4061] at e_99_2
  refine ⟨?_, ?_⟩
  · have hc := e_99_1
    simp only [e_98_11] at hc
    linear_combination c4065.trans k2 - hc
  · have hc := e_98_12
    simp only [e_98_10, e_99_2, e_98_11] at hc
    linear_combination c4066.trans k2 - hc

theorem privateBatchWrapper8_f477 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 26)) (a (.virt 70)) (a (.virt 543)) (a (.virt 544)) := by
  have c4067 := (privateBatchWrapper8_copies127 a h).2.2.2.1
  have c4068 := (privateBatchWrapper8_copies127 a h).2.2.2.2.1
  have c4069 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.1
  have c4070 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.1
  have c4071 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.1
  have c4072 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.1
  have c4073 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.1
  have c4074 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4075 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4076 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4077 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4078 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4079 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4080 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4081 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4082 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4083 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_98_13 := arithEq_of_rows h (row := 98) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_13
  simp only [← c4067, k3, ← c4068, k3, ← c4069] at e_98_13
  have e_98_14 := arithEq_of_rows h (row := 98) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_14
  simp only [← c4070, ← c4071, k3, ← c4072] at e_98_14
  have e_99_3 := arithEq_of_rows h (row := 99) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_3
  simp only [← c4073, ← c4074, ← c4075] at e_99_3
  have e_99_4 := arithEq_of_rows h (row := 99) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_4
  simp only [← c4076, ← c4077, ← c4078] at e_99_4
  have e_100_0 := arithEq_of_rows h (row := 100) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_0
  simp only [← c4079, ← c4080, k3, ← c4081] at e_100_0
  refine ⟨?_, ?_⟩
  · have hc := e_99_3
    simp only [e_98_14] at hc
    linear_combination c4082.trans k2 - hc
  · have hc := e_100_0
    simp only [e_98_13, e_99_4, e_98_14] at hc
    linear_combination c4083.trans k2 - hc

theorem privateBatchWrapper8_f478 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 27)) (a (.virt 71)) (a (.virt 545)) (a (.virt 546)) := by
  have c4084 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4085 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4086 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4087 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4088 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4089 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4090 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4091 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4092 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4093 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4094 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4095 := (privateBatchWrapper8_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4096 := (privateBatchWrapper8_copies128 a h).1
  have c4097 := (privateBatchWrapper8_copies128 a h).2.1
  have c4098 := (privateBatchWrapper8_copies128 a h).2.2.1
  have c4099 := (privateBatchWrapper8_copies128 a h).2.2.2.1
  have c4100 := (privateBatchWrapper8_copies128 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_99_5 := arithEq_of_rows h (row := 99) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_5
  simp only [← c4090, ← c4091, ← c4092] at e_99_5
  have e_99_6 := arithEq_of_rows h (row := 99) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_6
  simp only [← c4093, ← c4094, ← c4095] at e_99_6
  have e_100_1 := arithEq_of_rows h (row := 100) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_1
  simp only [← c4084, k3, ← c4085, k3, ← c4086] at e_100_1
  have e_100_2 := arithEq_of_rows h (row := 100) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_2
  simp only [← c4087, ← c4088, k3, ← c4089] at e_100_2
  have e_100_3 := arithEq_of_rows h (row := 100) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_3
  simp only [← c4096, ← c4097, k3, ← c4098] at e_100_3
  refine ⟨?_, ?_⟩
  · have hc := e_99_5
    simp only [e_100_2] at hc
    linear_combination c4099.trans k2 - hc
  · have hc := e_100_3
    simp only [e_100_1, e_99_6, e_100_2] at hc
    linear_combination c4100.trans k2 - hc

theorem privateBatchWrapper8_f479 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 28)) (a (.virt 72)) (a (.virt 547)) (a (.virt 548)) := by
  have c4101 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.1
  have c4102 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.1
  have c4103 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.1
  have c4104 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.1
  have c4105 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.1
  have c4106 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4107 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4108 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4109 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4110 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4111 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4112 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4113 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4114 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4115 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4116 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4117 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_99_7 := arithEq_of_rows h (row := 99) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_7
  simp only [← c4107, ← c4108, ← c4109] at e_99_7
  have e_99_8 := arithEq_of_rows h (row := 99) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_8
  simp only [← c4110, ← c4111, ← c4112] at e_99_8
  have e_100_4 := arithEq_of_rows h (row := 100) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_4
  simp only [← c4101, k3, ← c4102, k3, ← c4103] at e_100_4
  have e_100_5 := arithEq_of_rows h (row := 100) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_5
  simp only [← c4104, ← c4105, k3, ← c4106] at e_100_5
  have e_100_6 := arithEq_of_rows h (row := 100) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_6
  simp only [← c4113, ← c4114, k3, ← c4115] at e_100_6
  refine ⟨?_, ?_⟩
  · have hc := e_99_7
    simp only [e_100_5] at hc
    linear_combination c4116.trans k2 - hc
  · have hc := e_100_6
    simp only [e_100_4, e_99_8, e_100_5] at hc
    linear_combination c4117.trans k2 - hc

theorem privateBatchWrapper8_f480 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 39) = band (a (.virt 541)) (a (.virt 543)) := by
  have c4118 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4119 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4120 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_9 := arithEq_of_rows h (row := 99) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_9
  simp only [← c4118, ← c4119, ← c4120] at e_99_9
  have hr := e_99_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f481 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 43) = band (a (.virt 545)) (a (.virt 547)) := by
  have c4121 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4122 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4123 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_10 := arithEq_of_rows h (row := 99) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_10
  simp only [← c4121, ← c4122, ← c4123] at e_99_10
  have hr := e_99_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f482 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 47) = band (a (.wire 99 39)) (a (.wire 99 43)) := by
  have c4124 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4125 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4126 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_11 := arithEq_of_rows h (row := 99) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_11
  simp only [← c4124, ← c4125, ← c4126] at e_99_11
  have hr := e_99_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f483 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 51) = band (a (.wire 99 3)) (a (.wire 99 47)) := by
  have c4127 := (privateBatchWrapper8_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4128 := (privateBatchWrapper8_copies129 a h).1
  have c4129 := (privateBatchWrapper8_copies129 a h).2.1
  have e_99_12 := arithEq_of_rows h (row := 99) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_12
  simp only [← c4127, ← c4128, ← c4129] at e_99_12
  have hr := e_99_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f484 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 51) = a (.virt 11) := by
  have c4130 := (privateBatchWrapper8_copies129 a h).2.2.1
  exact c4130

theorem privateBatchWrapper8_f485 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f486 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 99 55) = band (a (.wire 13 31)) (a (.wire 20 31)) := by
  have c4131 := (privateBatchWrapper8_copies129 a h).2.2.2.1
  have c4132 := (privateBatchWrapper8_copies129 a h).2.2.2.2.1
  have c4133 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.1
  have e_99_13 := arithEq_of_rows h (row := 99) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_13
  simp only [← c4131, ← c4132, ← c4133] at e_99_13
  have hr := e_99_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f487 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 25)) (a (.virt 80)) (a (.virt 549)) (a (.virt 550)) := by
  have c4134 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.1
  have c4135 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.1
  have c4136 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.1
  have c4137 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.1
  have c4138 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4139 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4140 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4141 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4142 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4143 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4144 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4145 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4146 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4147 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4148 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4149 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4150 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_99_14 := arithEq_of_rows h (row := 99) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_14
  simp only [← c4140, ← c4141, ← c4142] at e_99_14
  have e_100_7 := arithEq_of_rows h (row := 100) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_7
  simp only [← c4134, k3, ← c4135, k3, ← c4136] at e_100_7
  have e_100_8 := arithEq_of_rows h (row := 100) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_8
  simp only [← c4137, ← c4138, k3, ← c4139] at e_100_8
  have e_100_9 := arithEq_of_rows h (row := 100) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_9
  simp only [← c4146, ← c4147, k3, ← c4148] at e_100_9
  have e_101_0 := arithEq_of_rows h (row := 101) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_0
  simp only [← c4143, ← c4144, ← c4145] at e_101_0
  refine ⟨?_, ?_⟩
  · have hc := e_99_14
    simp only [e_100_8] at hc
    linear_combination c4149.trans k2 - hc
  · have hc := e_100_9
    simp only [e_100_7, e_101_0, e_100_8] at hc
    linear_combination c4150.trans k2 - hc

theorem privateBatchWrapper8_f488 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 26)) (a (.virt 81)) (a (.virt 551)) (a (.virt 552)) := by
  have c4151 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4152 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4153 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4154 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4155 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4156 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4157 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4158 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4159 := (privateBatchWrapper8_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4160 := (privateBatchWrapper8_copies130 a h).1
  have c4161 := (privateBatchWrapper8_copies130 a h).2.1
  have c4162 := (privateBatchWrapper8_copies130 a h).2.2.1
  have c4163 := (privateBatchWrapper8_copies130 a h).2.2.2.1
  have c4164 := (privateBatchWrapper8_copies130 a h).2.2.2.2.1
  have c4165 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.1
  have c4166 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.1
  have c4167 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_100_10 := arithEq_of_rows h (row := 100) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_10
  simp only [← c4151, k3, ← c4152, k3, ← c4153] at e_100_10
  have e_100_11 := arithEq_of_rows h (row := 100) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_11
  simp only [← c4154, ← c4155, k3, ← c4156] at e_100_11
  have e_100_12 := arithEq_of_rows h (row := 100) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_12
  simp only [← c4163, ← c4164, k3, ← c4165] at e_100_12
  have e_101_1 := arithEq_of_rows h (row := 101) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_1
  simp only [← c4157, ← c4158, ← c4159] at e_101_1
  have e_101_2 := arithEq_of_rows h (row := 101) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_2
  simp only [← c4160, ← c4161, ← c4162] at e_101_2
  refine ⟨?_, ?_⟩
  · have hc := e_101_1
    simp only [e_100_11] at hc
    linear_combination c4166.trans k2 - hc
  · have hc := e_100_12
    simp only [e_100_10, e_101_2, e_100_11] at hc
    linear_combination c4167.trans k2 - hc

theorem privateBatchWrapper8_f489 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 27)) (a (.virt 82)) (a (.virt 553)) (a (.virt 554)) := by
  have c4168 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.1
  have c4169 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.1
  have c4170 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4171 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4172 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4173 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4174 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4175 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4176 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4177 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4178 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4179 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4180 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4181 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4182 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4183 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4184 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_100_13 := arithEq_of_rows h (row := 100) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_13
  simp only [← c4168, k3, ← c4169, k3, ← c4170] at e_100_13
  have e_100_14 := arithEq_of_rows h (row := 100) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_14
  simp only [← c4171, ← c4172, k3, ← c4173] at e_100_14
  have e_101_3 := arithEq_of_rows h (row := 101) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_3
  simp only [← c4174, ← c4175, ← c4176] at e_101_3
  have e_101_4 := arithEq_of_rows h (row := 101) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_4
  simp only [← c4177, ← c4178, ← c4179] at e_101_4
  have e_102_0 := arithEq_of_rows h (row := 102) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_0
  simp only [← c4180, ← c4181, k3, ← c4182] at e_102_0
  refine ⟨?_, ?_⟩
  · have hc := e_101_3
    simp only [e_100_14] at hc
    linear_combination c4183.trans k2 - hc
  · have hc := e_102_0
    simp only [e_100_13, e_101_4, e_100_14] at hc
    linear_combination c4184.trans k2 - hc

theorem privateBatchWrapper8_f490 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 28)) (a (.virt 83)) (a (.virt 555)) (a (.virt 556)) := by
  have c4185 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4186 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4187 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4188 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4189 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4190 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4191 := (privateBatchWrapper8_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4192 := (privateBatchWrapper8_copies131 a h).1
  have c4193 := (privateBatchWrapper8_copies131 a h).2.1
  have c4194 := (privateBatchWrapper8_copies131 a h).2.2.1
  have c4195 := (privateBatchWrapper8_copies131 a h).2.2.2.1
  have c4196 := (privateBatchWrapper8_copies131 a h).2.2.2.2.1
  have c4197 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.1
  have c4198 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.1
  have c4199 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.1
  have c4200 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.1
  have c4201 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_101_5 := arithEq_of_rows h (row := 101) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_5
  simp only [← c4191, ← c4192, ← c4193] at e_101_5
  have e_101_6 := arithEq_of_rows h (row := 101) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_6
  simp only [← c4194, ← c4195, ← c4196] at e_101_6
  have e_102_1 := arithEq_of_rows h (row := 102) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_1
  simp only [← c4185, k3, ← c4186, k3, ← c4187] at e_102_1
  have e_102_2 := arithEq_of_rows h (row := 102) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_2
  simp only [← c4188, ← c4189, k3, ← c4190] at e_102_2
  have e_102_3 := arithEq_of_rows h (row := 102) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_3
  simp only [← c4197, ← c4198, k3, ← c4199] at e_102_3
  refine ⟨?_, ?_⟩
  · have hc := e_101_5
    simp only [e_102_2] at hc
    linear_combination c4200.trans k2 - hc
  · have hc := e_102_3
    simp only [e_102_1, e_101_6, e_102_2] at hc
    linear_combination c4201.trans k2 - hc

theorem privateBatchWrapper8_f491 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 101 31) = band (a (.virt 549)) (a (.virt 551)) := by
  have c4202 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4203 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4204 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_7 := arithEq_of_rows h (row := 101) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_7
  simp only [← c4202, ← c4203, ← c4204] at e_101_7
  have hr := e_101_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f492 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 101 35) = band (a (.virt 553)) (a (.virt 555)) := by
  have c4205 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4206 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4207 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_8 := arithEq_of_rows h (row := 101) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_8
  simp only [← c4205, ← c4206, ← c4207] at e_101_8
  have hr := e_101_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f493 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 101 39) = band (a (.wire 101 31)) (a (.wire 101 35)) := by
  have c4208 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4209 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4210 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_9 := arithEq_of_rows h (row := 101) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_9
  simp only [← c4208, ← c4209, ← c4210] at e_101_9
  have hr := e_101_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f494 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 101 43) = band (a (.wire 99 55)) (a (.wire 101 39)) := by
  have c4211 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4212 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4213 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_10 := arithEq_of_rows h (row := 101) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_10
  simp only [← c4211, ← c4212, ← c4213] at e_101_10
  have hr := e_101_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f495 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 101 43) = a (.virt 11) := by
  have c4214 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4214

theorem privateBatchWrapper8_f496 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 15 19) = bnot (a (.wire 6 43)) := by
  have c628 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper8_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c628, k3, ← c629, k3, ← c630] at e_15_4
  have hr := e_15_4
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f497 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 7) = bnot (a (.wire 8 27)) := by
  have c673 := (privateBatchWrapper8_copies21 a h).2.1
  have c674 := (privateBatchWrapper8_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper8_copies21 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c673, k3, ← c674, k3, ← c675] at e_17_1
  have hr := e_17_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f498 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 101 47) = band (a (.wire 15 19)) (a (.wire 17 7)) := by
  have c4215 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4216 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4217 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_11 := arithEq_of_rows h (row := 101) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_11
  simp only [← c4215, ← c4216, ← c4217] at e_101_11
  have hr := e_101_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f499 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 36)) (a (.virt 47)) (a (.virt 557)) (a (.virt 558)) := by
  have c4218 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4219 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4220 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4221 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4222 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4223 := (privateBatchWrapper8_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4224 := (privateBatchWrapper8_copies132 a h).1
  have c4225 := (privateBatchWrapper8_copies132 a h).2.1
  have c4226 := (privateBatchWrapper8_copies132 a h).2.2.1
  have c4227 := (privateBatchWrapper8_copies132 a h).2.2.2.1
  have c4228 := (privateBatchWrapper8_copies132 a h).2.2.2.2.1
  have c4229 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.1
  have c4230 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.1
  have c4231 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.1
  have c4232 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.1
  have c4233 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.1
  have c4234 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_101_12 := arithEq_of_rows h (row := 101) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_12
  simp only [← c4224, ← c4225, ← c4226] at e_101_12
  have e_101_13 := arithEq_of_rows h (row := 101) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_13
  simp only [← c4227, ← c4228, ← c4229] at e_101_13
  have e_102_4 := arithEq_of_rows h (row := 102) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_4
  simp only [← c4218, k3, ← c4219, k3, ← c4220] at e_102_4
  have e_102_5 := arithEq_of_rows h (row := 102) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_5
  simp only [← c4221, ← c4222, k3, ← c4223] at e_102_5
  have e_102_6 := arithEq_of_rows h (row := 102) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_6
  simp only [← c4230, ← c4231, k3, ← c4232] at e_102_6
  refine ⟨?_, ?_⟩
  · have hc := e_101_12
    simp only [e_102_5] at hc
    linear_combination c4233.trans k2 - hc
  · have hc := e_102_6
    simp only [e_102_4, e_101_13, e_102_5] at hc
    linear_combination c4234.trans k2 - hc

theorem privateBatchWrapper8_f500 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 37)) (a (.virt 48)) (a (.virt 559)) (a (.virt 560)) := by
  have c4235 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4236 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4237 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4238 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4239 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4240 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4241 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4242 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4243 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4244 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4245 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4246 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4247 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4248 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4249 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4250 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4251 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_101_14 := arithEq_of_rows h (row := 101) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_14
  simp only [← c4241, ← c4242, ← c4243] at e_101_14
  have e_102_7 := arithEq_of_rows h (row := 102) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_7
  simp only [← c4235, k3, ← c4236, k3, ← c4237] at e_102_7
  have e_102_8 := arithEq_of_rows h (row := 102) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_8
  simp only [← c4238, ← c4239, k3, ← c4240] at e_102_8
  have e_102_9 := arithEq_of_rows h (row := 102) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_9
  simp only [← c4247, ← c4248, k3, ← c4249] at e_102_9
  have e_103_0 := arithEq_of_rows h (row := 103) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_0
  simp only [← c4244, ← c4245, ← c4246] at e_103_0
  refine ⟨?_, ?_⟩
  · have hc := e_101_14
    simp only [e_102_8] at hc
    linear_combination c4250.trans k2 - hc
  · have hc := e_102_9
    simp only [e_102_7, e_103_0, e_102_8] at hc
    linear_combination c4251.trans k2 - hc

theorem privateBatchWrapper8_f501 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 38)) (a (.virt 49)) (a (.virt 561)) (a (.virt 562)) := by
  have c4252 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4253 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4254 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4255 := (privateBatchWrapper8_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4256 := (privateBatchWrapper8_copies133 a h).1
  have c4257 := (privateBatchWrapper8_copies133 a h).2.1
  have c4258 := (privateBatchWrapper8_copies133 a h).2.2.1
  have c4259 := (privateBatchWrapper8_copies133 a h).2.2.2.1
  have c4260 := (privateBatchWrapper8_copies133 a h).2.2.2.2.1
  have c4261 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.1
  have c4262 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.1
  have c4263 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.1
  have c4264 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.1
  have c4265 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.1
  have c4266 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4267 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4268 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_102_10 := arithEq_of_rows h (row := 102) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_10
  simp only [← c4252, k3, ← c4253, k3, ← c4254] at e_102_10
  have e_102_11 := arithEq_of_rows h (row := 102) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_11
  simp only [← c4255, ← c4256, k3, ← c4257] at e_102_11
  have e_102_12 := arithEq_of_rows h (row := 102) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_12
  simp only [← c4264, ← c4265, k3, ← c4266] at e_102_12
  have e_103_1 := arithEq_of_rows h (row := 103) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_1
  simp only [← c4258, ← c4259, ← c4260] at e_103_1
  have e_103_2 := arithEq_of_rows h (row := 103) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_2
  simp only [← c4261, ← c4262, ← c4263] at e_103_2
  refine ⟨?_, ?_⟩
  · have hc := e_103_1
    simp only [e_102_11] at hc
    linear_combination c4267.trans k2 - hc
  · have hc := e_102_12
    simp only [e_102_10, e_103_2, e_102_11] at hc
    linear_combination c4268.trans k2 - hc

theorem privateBatchWrapper8_f502 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 39)) (a (.virt 50)) (a (.virt 563)) (a (.virt 564)) := by
  have c4269 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4270 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4271 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4272 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4273 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4274 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4275 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4276 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4277 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4278 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4279 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4280 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4281 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4282 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4283 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4284 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4285 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_102_13 := arithEq_of_rows h (row := 102) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_13
  simp only [← c4269, k3, ← c4270, k3, ← c4271] at e_102_13
  have e_102_14 := arithEq_of_rows h (row := 102) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_14
  simp only [← c4272, ← c4273, k3, ← c4274] at e_102_14
  have e_103_3 := arithEq_of_rows h (row := 103) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_3
  simp only [← c4275, ← c4276, ← c4277] at e_103_3
  have e_103_4 := arithEq_of_rows h (row := 103) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_4
  simp only [← c4278, ← c4279, ← c4280] at e_103_4
  have e_104_0 := arithEq_of_rows h (row := 104) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c4281, ← c4282, k3, ← c4283] at e_104_0
  refine ⟨?_, ?_⟩
  · have hc := e_103_3
    simp only [e_102_14] at hc
    linear_combination c4284.trans k2 - hc
  · have hc := e_104_0
    simp only [e_102_13, e_103_4, e_102_14] at hc
    linear_combination c4285.trans k2 - hc

theorem privateBatchWrapper8_f503 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 103 23) = band (a (.virt 557)) (a (.virt 559)) := by
  have c4286 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4287 := (privateBatchWrapper8_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4288 := (privateBatchWrapper8_copies134 a h).1
  have e_103_5 := arithEq_of_rows h (row := 103) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_5
  simp only [← c4286, ← c4287, ← c4288] at e_103_5
  have hr := e_103_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f504 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 103 27) = band (a (.virt 561)) (a (.virt 563)) := by
  have c4289 := (privateBatchWrapper8_copies134 a h).2.1
  have c4290 := (privateBatchWrapper8_copies134 a h).2.2.1
  have c4291 := (privateBatchWrapper8_copies134 a h).2.2.2.1
  have e_103_6 := arithEq_of_rows h (row := 103) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_6
  simp only [← c4289, ← c4290, ← c4291] at e_103_6
  have hr := e_103_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f505 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 103 31) = band (a (.wire 103 23)) (a (.wire 103 27)) := by
  have c4292 := (privateBatchWrapper8_copies134 a h).2.2.2.2.1
  have c4293 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.1
  have c4294 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.1
  have e_103_7 := arithEq_of_rows h (row := 103) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_7
  simp only [← c4292, ← c4293, ← c4294] at e_103_7
  have hr := e_103_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f506 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 103 35) = band (a (.wire 101 47)) (a (.wire 103 31)) := by
  have c4295 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.1
  have c4296 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.1
  have c4297 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.1
  have e_103_8 := arithEq_of_rows h (row := 103) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_8
  simp only [← c4295, ← c4296, ← c4297] at e_103_8
  have hr := e_103_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f507 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 103 35) = a (.virt 11) := by
  have c4298 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c4298

theorem privateBatchWrapper8_f508 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f509 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 103 39) = band (a (.wire 15 19)) (a (.wire 17 55)) := by
  have c4299 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4300 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4301 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_9 := arithEq_of_rows h (row := 103) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_9
  simp only [← c4299, ← c4300, ← c4301] at e_103_9
  have hr := e_103_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f510 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 36)) (a (.virt 58)) (a (.virt 565)) (a (.virt 566)) := by
  have c4302 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4303 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4304 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4305 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4306 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4307 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4308 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4309 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4310 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4311 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4312 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4313 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4314 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4315 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4316 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4317 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4318 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_103_10 := arithEq_of_rows h (row := 103) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_10
  simp only [← c4308, ← c4309, ← c4310] at e_103_10
  have e_103_11 := arithEq_of_rows h (row := 103) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_11
  simp only [← c4311, ← c4312, ← c4313] at e_103_11
  have e_104_1 := arithEq_of_rows h (row := 104) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_1
  simp only [← c4302, k3, ← c4303, k3, ← c4304] at e_104_1
  have e_104_2 := arithEq_of_rows h (row := 104) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_2
  simp only [← c4305, ← c4306, k3, ← c4307] at e_104_2
  have e_104_3 := arithEq_of_rows h (row := 104) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_3
  simp only [← c4314, ← c4315, k3, ← c4316] at e_104_3
  refine ⟨?_, ?_⟩
  · have hc := e_103_10
    simp only [e_104_2] at hc
    linear_combination c4317.trans k2 - hc
  · have hc := e_104_3
    simp only [e_104_1, e_103_11, e_104_2] at hc
    linear_combination c4318.trans k2 - hc

theorem privateBatchWrapper8_f511 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 37)) (a (.virt 59)) (a (.virt 567)) (a (.virt 568)) := by
  have c4319 := (privateBatchWrapper8_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4320 := (privateBatchWrapper8_copies135 a h).1
  have c4321 := (privateBatchWrapper8_copies135 a h).2.1
  have c4322 := (privateBatchWrapper8_copies135 a h).2.2.1
  have c4323 := (privateBatchWrapper8_copies135 a h).2.2.2.1
  have c4324 := (privateBatchWrapper8_copies135 a h).2.2.2.2.1
  have c4325 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.1
  have c4326 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.1
  have c4327 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.1
  have c4328 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.1
  have c4329 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.1
  have c4330 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4331 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4332 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4333 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4334 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4335 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_103_12 := arithEq_of_rows h (row := 103) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_12
  simp only [← c4325, ← c4326, ← c4327] at e_103_12
  have e_103_13 := arithEq_of_rows h (row := 103) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_13
  simp only [← c4328, ← c4329, ← c4330] at e_103_13
  have e_104_4 := arithEq_of_rows h (row := 104) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c4319, k3, ← c4320, k3, ← c4321] at e_104_4
  have e_104_5 := arithEq_of_rows h (row := 104) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_5
  simp only [← c4322, ← c4323, k3, ← c4324] at e_104_5
  have e_104_6 := arithEq_of_rows h (row := 104) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_6
  simp only [← c4331, ← c4332, k3, ← c4333] at e_104_6
  refine ⟨?_, ?_⟩
  · have hc := e_103_12
    simp only [e_104_5] at hc
    linear_combination c4334.trans k2 - hc
  · have hc := e_104_6
    simp only [e_104_4, e_103_13, e_104_5] at hc
    linear_combination c4335.trans k2 - hc

theorem privateBatchWrapper8_f512 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 38)) (a (.virt 60)) (a (.virt 569)) (a (.virt 570)) := by
  have c4336 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4337 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4338 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4339 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4340 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4341 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4342 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4343 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4344 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4345 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4346 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4347 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4348 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4349 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4350 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4351 := (privateBatchWrapper8_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4352 := (privateBatchWrapper8_copies136 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_103_14 := arithEq_of_rows h (row := 103) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_14
  simp only [← c4342, ← c4343, ← c4344] at e_103_14
  have e_104_7 := arithEq_of_rows h (row := 104) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c4336, k3, ← c4337, k3, ← c4338] at e_104_7
  have e_104_8 := arithEq_of_rows h (row := 104) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_8
  simp only [← c4339, ← c4340, k3, ← c4341] at e_104_8
  have e_104_9 := arithEq_of_rows h (row := 104) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_9
  simp only [← c4348, ← c4349, k3, ← c4350] at e_104_9
  have e_105_0 := arithEq_of_rows h (row := 105) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_0
  simp only [← c4345, ← c4346, ← c4347] at e_105_0
  refine ⟨?_, ?_⟩
  · have hc := e_103_14
    simp only [e_104_8] at hc
    linear_combination c4351.trans k2 - hc
  · have hc := e_104_9
    simp only [e_104_7, e_105_0, e_104_8] at hc
    linear_combination c4352.trans k2 - hc

theorem privateBatchWrapper8_f513 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 39)) (a (.virt 61)) (a (.virt 571)) (a (.virt 572)) := by
  have c4353 := (privateBatchWrapper8_copies136 a h).2.1
  have c4354 := (privateBatchWrapper8_copies136 a h).2.2.1
  have c4355 := (privateBatchWrapper8_copies136 a h).2.2.2.1
  have c4356 := (privateBatchWrapper8_copies136 a h).2.2.2.2.1
  have c4357 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.1
  have c4358 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.1
  have c4359 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.1
  have c4360 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.1
  have c4361 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.1
  have c4362 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4363 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4364 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4365 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4366 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4367 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4368 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4369 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c4353, k3, ← c4354, k3, ← c4355] at e_104_10
  have e_104_11 := arithEq_of_rows h (row := 104) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_11
  simp only [← c4356, ← c4357, k3, ← c4358] at e_104_11
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c4365, ← c4366, k3, ← c4367] at e_104_12
  have e_105_1 := arithEq_of_rows h (row := 105) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_1
  simp only [← c4359, ← c4360, ← c4361] at e_105_1
  have e_105_2 := arithEq_of_rows h (row := 105) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_2
  simp only [← c4362, ← c4363, ← c4364] at e_105_2
  refine ⟨?_, ?_⟩
  · have hc := e_105_1
    simp only [e_104_11] at hc
    linear_combination c4368.trans k2 - hc
  · have hc := e_104_12
    simp only [e_104_10, e_105_2, e_104_11] at hc
    linear_combination c4369.trans k2 - hc

theorem privateBatchWrapper8_f514 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 105 15) = band (a (.virt 565)) (a (.virt 567)) := by
  have c4370 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4371 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4372 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_3 := arithEq_of_rows h (row := 105) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_3
  simp only [← c4370, ← c4371, ← c4372] at e_105_3
  have hr := e_105_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f515 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 105 19) = band (a (.virt 569)) (a (.virt 571)) := by
  have c4373 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4374 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4375 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_4 := arithEq_of_rows h (row := 105) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_4
  simp only [← c4373, ← c4374, ← c4375] at e_105_4
  have hr := e_105_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f516 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 105 23) = band (a (.wire 105 15)) (a (.wire 105 19)) := by
  have c4376 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4377 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4378 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_5 := arithEq_of_rows h (row := 105) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_5
  simp only [← c4376, ← c4377, ← c4378] at e_105_5
  have hr := e_105_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f517 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 105 27) = band (a (.wire 103 39)) (a (.wire 105 23)) := by
  have c4379 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4380 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4381 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_6 := arithEq_of_rows h (row := 105) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_6
  simp only [← c4379, ← c4380, ← c4381] at e_105_6
  have hr := e_105_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f518 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 105 27) = a (.virt 11) := by
  have c4382 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4382

theorem privateBatchWrapper8_f519 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f520 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 105 31) = band (a (.wire 15 19)) (a (.wire 18 43)) := by
  have c4383 := (privateBatchWrapper8_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4384 := (privateBatchWrapper8_copies137 a h).1
  have c4385 := (privateBatchWrapper8_copies137 a h).2.1
  have e_105_7 := arithEq_of_rows h (row := 105) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_7
  simp only [← c4383, ← c4384, ← c4385] at e_105_7
  have hr := e_105_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f521 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 36)) (a (.virt 69)) (a (.virt 573)) (a (.virt 574)) := by
  have c4386 := (privateBatchWrapper8_copies137 a h).2.2.1
  have c4387 := (privateBatchWrapper8_copies137 a h).2.2.2.1
  have c4388 := (privateBatchWrapper8_copies137 a h).2.2.2.2.1
  have c4389 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.1
  have c4390 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.1
  have c4391 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.1
  have c4392 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.1
  have c4393 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.1
  have c4394 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4395 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4396 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4397 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4398 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4399 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4400 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4401 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4402 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_104_13 := arithEq_of_rows h (row := 104) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_13
  simp only [← c4386, k3, ← c4387, k3, ← c4388] at e_104_13
  have e_104_14 := arithEq_of_rows h (row := 104) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_14
  simp only [← c4389, ← c4390, k3, ← c4391] at e_104_14
  have e_105_8 := arithEq_of_rows h (row := 105) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_8
  simp only [← c4392, ← c4393, ← c4394] at e_105_8
  have e_105_9 := arithEq_of_rows h (row := 105) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_9
  simp only [← c4395, ← c4396, ← c4397] at e_105_9
  have e_106_0 := arithEq_of_rows h (row := 106) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_0
  simp only [← c4398, ← c4399, k3, ← c4400] at e_106_0
  refine ⟨?_, ?_⟩
  · have hc := e_105_8
    simp only [e_104_14] at hc
    linear_combination c4401.trans k2 - hc
  · have hc := e_106_0
    simp only [e_104_13, e_105_9, e_104_14] at hc
    linear_combination c4402.trans k2 - hc

theorem privateBatchWrapper8_f522 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 37)) (a (.virt 70)) (a (.virt 575)) (a (.virt 576)) := by
  have c4403 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4404 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4405 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4406 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4407 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4408 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4409 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4410 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4411 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4412 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4413 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4414 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4415 := (privateBatchWrapper8_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4416 := (privateBatchWrapper8_copies138 a h).1
  have c4417 := (privateBatchWrapper8_copies138 a h).2.1
  have c4418 := (privateBatchWrapper8_copies138 a h).2.2.1
  have c4419 := (privateBatchWrapper8_copies138 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_105_10 := arithEq_of_rows h (row := 105) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_10
  simp only [← c4409, ← c4410, ← c4411] at e_105_10
  have e_105_11 := arithEq_of_rows h (row := 105) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_11
  simp only [← c4412, ← c4413, ← c4414] at e_105_11
  have e_106_1 := arithEq_of_rows h (row := 106) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_1
  simp only [← c4403, k3, ← c4404, k3, ← c4405] at e_106_1
  have e_106_2 := arithEq_of_rows h (row := 106) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_2
  simp only [← c4406, ← c4407, k3, ← c4408] at e_106_2
  have e_106_3 := arithEq_of_rows h (row := 106) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_3
  simp only [← c4415, ← c4416, k3, ← c4417] at e_106_3
  refine ⟨?_, ?_⟩
  · have hc := e_105_10
    simp only [e_106_2] at hc
    linear_combination c4418.trans k2 - hc
  · have hc := e_106_3
    simp only [e_106_1, e_105_11, e_106_2] at hc
    linear_combination c4419.trans k2 - hc

theorem privateBatchWrapper8_f523 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 38)) (a (.virt 71)) (a (.virt 577)) (a (.virt 578)) := by
  have c4420 := (privateBatchWrapper8_copies138 a h).2.2.2.2.1
  have c4421 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.1
  have c4422 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.1
  have c4423 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.1
  have c4424 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.1
  have c4425 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.1
  have c4426 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4427 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4428 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4429 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4430 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4432 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4435 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4436 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_105_12 := arithEq_of_rows h (row := 105) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_12
  simp only [← c4426, ← c4427, ← c4428] at e_105_12
  have e_105_13 := arithEq_of_rows h (row := 105) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_13
  simp only [← c4429, ← c4430, ← c4431] at e_105_13
  have e_106_4 := arithEq_of_rows h (row := 106) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_4
  simp only [← c4420, k3, ← c4421, k3, ← c4422] at e_106_4
  have e_106_5 := arithEq_of_rows h (row := 106) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_5
  simp only [← c4423, ← c4424, k3, ← c4425] at e_106_5
  have e_106_6 := arithEq_of_rows h (row := 106) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_6
  simp only [← c4432, ← c4433, k3, ← c4434] at e_106_6
  refine ⟨?_, ?_⟩
  · have hc := e_105_12
    simp only [e_106_5] at hc
    linear_combination c4435.trans k2 - hc
  · have hc := e_106_6
    simp only [e_106_4, e_105_13, e_106_5] at hc
    linear_combination c4436.trans k2 - hc

theorem privateBatchWrapper8_f524 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 39)) (a (.virt 72)) (a (.virt 579)) (a (.virt 580)) := by
  have c4437 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4438 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4439 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4440 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4441 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4442 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4443 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4444 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4445 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4446 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4447 := (privateBatchWrapper8_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4448 := (privateBatchWrapper8_copies139 a h).1
  have c4449 := (privateBatchWrapper8_copies139 a h).2.1
  have c4450 := (privateBatchWrapper8_copies139 a h).2.2.1
  have c4451 := (privateBatchWrapper8_copies139 a h).2.2.2.1
  have c4452 := (privateBatchWrapper8_copies139 a h).2.2.2.2.1
  have c4453 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_105_14 := arithEq_of_rows h (row := 105) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_14
  simp only [← c4443, ← c4444, ← c4445] at e_105_14
  have e_106_7 := arithEq_of_rows h (row := 106) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_7
  simp only [← c4437, k3, ← c4438, k3, ← c4439] at e_106_7
  have e_106_8 := arithEq_of_rows h (row := 106) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_8
  simp only [← c4440, ← c4441, k3, ← c4442] at e_106_8
  have e_106_9 := arithEq_of_rows h (row := 106) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_9
  simp only [← c4449, ← c4450, k3, ← c4451] at e_106_9
  have e_107_0 := arithEq_of_rows h (row := 107) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_0
  simp only [← c4446, ← c4447, ← c4448] at e_107_0
  refine ⟨?_, ?_⟩
  · have hc := e_105_14
    simp only [e_106_8] at hc
    linear_combination c4452.trans k2 - hc
  · have hc := e_106_9
    simp only [e_106_7, e_107_0, e_106_8] at hc
    linear_combination c4453.trans k2 - hc

theorem privateBatchWrapper8_f525 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 7) = band (a (.virt 573)) (a (.virt 575)) := by
  have c4454 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.1
  have c4455 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.1
  have c4456 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.1
  have e_107_1 := arithEq_of_rows h (row := 107) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_1
  simp only [← c4454, ← c4455, ← c4456] at e_107_1
  have hr := e_107_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f526 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 11) = band (a (.virt 577)) (a (.virt 579)) := by
  have c4457 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.1
  have c4458 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4459 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_2 := arithEq_of_rows h (row := 107) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_2
  simp only [← c4457, ← c4458, ← c4459] at e_107_2
  have hr := e_107_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f527 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 15) = band (a (.wire 107 7)) (a (.wire 107 11)) := by
  have c4460 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4461 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4462 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_3 := arithEq_of_rows h (row := 107) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_3
  simp only [← c4460, ← c4461, ← c4462] at e_107_3
  have hr := e_107_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f528 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 19) = band (a (.wire 105 31)) (a (.wire 107 15)) := by
  have c4463 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4464 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4465 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_4 := arithEq_of_rows h (row := 107) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_4
  simp only [← c4463, ← c4464, ← c4465] at e_107_4
  have hr := e_107_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f529 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 19) = a (.virt 11) := by
  have c4466 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4466

theorem privateBatchWrapper8_f530 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f531 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 23) = band (a (.wire 15 19)) (a (.wire 20 31)) := by
  have c4467 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4468 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4469 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_5 := arithEq_of_rows h (row := 107) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_5
  simp only [← c4467, ← c4468, ← c4469] at e_107_5
  have hr := e_107_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f532 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 36)) (a (.virt 80)) (a (.virt 581)) (a (.virt 582)) := by
  have c4470 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4471 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4472 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4473 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4474 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4475 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4476 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4477 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4478 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4479 := (privateBatchWrapper8_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4480 := (privateBatchWrapper8_copies140 a h).1
  have c4481 := (privateBatchWrapper8_copies140 a h).2.1
  have c4482 := (privateBatchWrapper8_copies140 a h).2.2.1
  have c4483 := (privateBatchWrapper8_copies140 a h).2.2.2.1
  have c4484 := (privateBatchWrapper8_copies140 a h).2.2.2.2.1
  have c4485 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.1
  have c4486 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_106_10 := arithEq_of_rows h (row := 106) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_10
  simp only [← c4470, k3, ← c4471, k3, ← c4472] at e_106_10
  have e_106_11 := arithEq_of_rows h (row := 106) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_11
  simp only [← c4473, ← c4474, k3, ← c4475] at e_106_11
  have e_106_12 := arithEq_of_rows h (row := 106) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_12
  simp only [← c4482, ← c4483, k3, ← c4484] at e_106_12
  have e_107_6 := arithEq_of_rows h (row := 107) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_6
  simp only [← c4476, ← c4477, ← c4478] at e_107_6
  have e_107_7 := arithEq_of_rows h (row := 107) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_7
  simp only [← c4479, ← c4480, ← c4481] at e_107_7
  refine ⟨?_, ?_⟩
  · have hc := e_107_6
    simp only [e_106_11] at hc
    linear_combination c4485.trans k2 - hc
  · have hc := e_106_12
    simp only [e_106_10, e_107_7, e_106_11] at hc
    linear_combination c4486.trans k2 - hc

theorem privateBatchWrapper8_f533 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 37)) (a (.virt 81)) (a (.virt 583)) (a (.virt 584)) := by
  have c4487 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.1
  have c4488 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.1
  have c4489 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.1
  have c4490 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4491 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4492 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4493 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4494 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4495 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4496 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4497 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4498 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4499 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4500 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4501 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4502 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4503 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_106_13 := arithEq_of_rows h (row := 106) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_13
  simp only [← c4487, k3, ← c4488, k3, ← c4489] at e_106_13
  have e_106_14 := arithEq_of_rows h (row := 106) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_14
  simp only [← c4490, ← c4491, k3, ← c4492] at e_106_14
  have e_107_8 := arithEq_of_rows h (row := 107) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_8
  simp only [← c4493, ← c4494, ← c4495] at e_107_8
  have e_107_9 := arithEq_of_rows h (row := 107) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_9
  simp only [← c4496, ← c4497, ← c4498] at e_107_9
  have e_108_0 := arithEq_of_rows h (row := 108) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_0
  simp only [← c4499, ← c4500, k3, ← c4501] at e_108_0
  refine ⟨?_, ?_⟩
  · have hc := e_107_8
    simp only [e_106_14] at hc
    linear_combination c4502.trans k2 - hc
  · have hc := e_108_0
    simp only [e_106_13, e_107_9, e_106_14] at hc
    linear_combination c4503.trans k2 - hc

theorem privateBatchWrapper8_f534 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 38)) (a (.virt 82)) (a (.virt 585)) (a (.virt 586)) := by
  have c4504 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4505 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4506 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4507 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4508 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4509 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4510 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4511 := (privateBatchWrapper8_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4512 := (privateBatchWrapper8_copies141 a h).1
  have c4513 := (privateBatchWrapper8_copies141 a h).2.1
  have c4514 := (privateBatchWrapper8_copies141 a h).2.2.1
  have c4515 := (privateBatchWrapper8_copies141 a h).2.2.2.1
  have c4516 := (privateBatchWrapper8_copies141 a h).2.2.2.2.1
  have c4517 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.1
  have c4518 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.1
  have c4519 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.1
  have c4520 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_107_10 := arithEq_of_rows h (row := 107) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_10
  simp only [← c4510, ← c4511, ← c4512] at e_107_10
  have e_107_11 := arithEq_of_rows h (row := 107) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_11
  simp only [← c4513, ← c4514, ← c4515] at e_107_11
  have e_108_1 := arithEq_of_rows h (row := 108) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_1
  simp only [← c4504, k3, ← c4505, k3, ← c4506] at e_108_1
  have e_108_2 := arithEq_of_rows h (row := 108) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_2
  simp only [← c4507, ← c4508, k3, ← c4509] at e_108_2
  have e_108_3 := arithEq_of_rows h (row := 108) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_3
  simp only [← c4516, ← c4517, k3, ← c4518] at e_108_3
  refine ⟨?_, ?_⟩
  · have hc := e_107_10
    simp only [e_108_2] at hc
    linear_combination c4519.trans k2 - hc
  · have hc := e_108_3
    simp only [e_108_1, e_107_11, e_108_2] at hc
    linear_combination c4520.trans k2 - hc

theorem privateBatchWrapper8_f535 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 39)) (a (.virt 83)) (a (.virt 587)) (a (.virt 588)) := by
  have c4521 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.1
  have c4522 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4523 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4524 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4525 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4526 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4527 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4528 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4529 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4530 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4531 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4532 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4533 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4534 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4535 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4536 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4537 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_107_12 := arithEq_of_rows h (row := 107) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_12
  simp only [← c4527, ← c4528, ← c4529] at e_107_12
  have e_107_13 := arithEq_of_rows h (row := 107) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_13
  simp only [← c4530, ← c4531, ← c4532] at e_107_13
  have e_108_4 := arithEq_of_rows h (row := 108) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_4
  simp only [← c4521, k3, ← c4522, k3, ← c4523] at e_108_4
  have e_108_5 := arithEq_of_rows h (row := 108) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_5
  simp only [← c4524, ← c4525, k3, ← c4526] at e_108_5
  have e_108_6 := arithEq_of_rows h (row := 108) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_6
  simp only [← c4533, ← c4534, k3, ← c4535] at e_108_6
  refine ⟨?_, ?_⟩
  · have hc := e_107_12
    simp only [e_108_5] at hc
    linear_combination c4536.trans k2 - hc
  · have hc := e_108_6
    simp only [e_108_4, e_107_13, e_108_5] at hc
    linear_combination c4537.trans k2 - hc

theorem privateBatchWrapper8_f536 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 107 59) = band (a (.virt 581)) (a (.virt 583)) := by
  have c4538 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4539 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4540 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_14 := arithEq_of_rows h (row := 107) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_14
  simp only [← c4538, ← c4539, ← c4540] at e_107_14
  have hr := e_107_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f537 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 3) = band (a (.virt 585)) (a (.virt 587)) := by
  have c4541 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4542 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4543 := (privateBatchWrapper8_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_109_0 := arithEq_of_rows h (row := 109) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_0
  simp only [← c4541, ← c4542, ← c4543] at e_109_0
  have hr := e_109_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f538 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 7) = band (a (.wire 107 59)) (a (.wire 109 3)) := by
  have c4544 := (privateBatchWrapper8_copies142 a h).1
  have c4545 := (privateBatchWrapper8_copies142 a h).2.1
  have c4546 := (privateBatchWrapper8_copies142 a h).2.2.1
  have e_109_1 := arithEq_of_rows h (row := 109) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_1
  simp only [← c4544, ← c4545, ← c4546] at e_109_1
  have hr := e_109_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f539 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 11) = band (a (.wire 107 23)) (a (.wire 109 7)) := by
  have c4547 := (privateBatchWrapper8_copies142 a h).2.2.2.1
  have c4548 := (privateBatchWrapper8_copies142 a h).2.2.2.2.1
  have c4549 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.1
  have e_109_2 := arithEq_of_rows h (row := 109) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_2
  simp only [← c4547, ← c4548, ← c4549] at e_109_2
  have hr := e_109_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f540 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 11) = a (.virt 11) := by
  have c4550 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.1
  exact c4550

theorem privateBatchWrapper8_f541 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 7) = bnot (a (.wire 8 27)) := by
  have c673 := (privateBatchWrapper8_copies21 a h).2.1
  have c674 := (privateBatchWrapper8_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper8_copies21 a h).2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c673, k3, ← c674, k3, ← c675] at e_17_1
  have hr := e_17_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f542 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f543 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 15) = band (a (.wire 17 7)) (a (.wire 17 55)) := by
  have c4551 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.1
  have c4552 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.1
  have c4553 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.1
  have e_109_3 := arithEq_of_rows h (row := 109) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_3
  simp only [← c4551, ← c4552, ← c4553] at e_109_3
  have hr := e_109_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f544 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 47)) (a (.virt 58)) (a (.virt 589)) (a (.virt 590)) := by
  have c4554 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4555 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4556 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4557 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4558 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4559 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4560 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4561 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4562 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4563 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4564 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4565 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4566 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4567 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4568 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4569 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4570 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_108_7 := arithEq_of_rows h (row := 108) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_7
  simp only [← c4554, k3, ← c4555, k3, ← c4556] at e_108_7
  have e_108_8 := arithEq_of_rows h (row := 108) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_8
  simp only [← c4557, ← c4558, k3, ← c4559] at e_108_8
  have e_108_9 := arithEq_of_rows h (row := 108) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_9
  simp only [← c4566, ← c4567, k3, ← c4568] at e_108_9
  have e_109_4 := arithEq_of_rows h (row := 109) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_4
  simp only [← c4560, ← c4561, ← c4562] at e_109_4
  have e_109_5 := arithEq_of_rows h (row := 109) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_5
  simp only [← c4563, ← c4564, ← c4565] at e_109_5
  refine ⟨?_, ?_⟩
  · have hc := e_109_4
    simp only [e_108_8] at hc
    linear_combination c4569.trans k2 - hc
  · have hc := e_108_9
    simp only [e_108_7, e_109_5, e_108_8] at hc
    linear_combination c4570.trans k2 - hc

theorem privateBatchWrapper8_f545 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 48)) (a (.virt 59)) (a (.virt 591)) (a (.virt 592)) := by
  have c4571 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4572 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4573 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4574 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4575 := (privateBatchWrapper8_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4576 := (privateBatchWrapper8_copies143 a h).1
  have c4577 := (privateBatchWrapper8_copies143 a h).2.1
  have c4578 := (privateBatchWrapper8_copies143 a h).2.2.1
  have c4579 := (privateBatchWrapper8_copies143 a h).2.2.2.1
  have c4580 := (privateBatchWrapper8_copies143 a h).2.2.2.2.1
  have c4581 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.1
  have c4582 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.1
  have c4583 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.1
  have c4584 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.1
  have c4585 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.1
  have c4586 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4587 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_108_10 := arithEq_of_rows h (row := 108) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_10
  simp only [← c4571, k3, ← c4572, k3, ← c4573] at e_108_10
  have e_108_11 := arithEq_of_rows h (row := 108) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_11
  simp only [← c4574, ← c4575, k3, ← c4576] at e_108_11
  have e_108_12 := arithEq_of_rows h (row := 108) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_12
  simp only [← c4583, ← c4584, k3, ← c4585] at e_108_12
  have e_109_6 := arithEq_of_rows h (row := 109) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_6
  simp only [← c4577, ← c4578, ← c4579] at e_109_6
  have e_109_7 := arithEq_of_rows h (row := 109) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_7
  simp only [← c4580, ← c4581, ← c4582] at e_109_7
  refine ⟨?_, ?_⟩
  · have hc := e_109_6
    simp only [e_108_11] at hc
    linear_combination c4586.trans k2 - hc
  · have hc := e_108_12
    simp only [e_108_10, e_109_7, e_108_11] at hc
    linear_combination c4587.trans k2 - hc

theorem privateBatchWrapper8_f546 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 49)) (a (.virt 60)) (a (.virt 593)) (a (.virt 594)) := by
  have c4588 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4589 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4590 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4591 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4592 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4593 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4594 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4595 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4596 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4597 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4598 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4599 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4600 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4601 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4602 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4603 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4604 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_108_13 := arithEq_of_rows h (row := 108) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_13
  simp only [← c4588, k3, ← c4589, k3, ← c4590] at e_108_13
  have e_108_14 := arithEq_of_rows h (row := 108) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_14
  simp only [← c4591, ← c4592, k3, ← c4593] at e_108_14
  have e_109_8 := arithEq_of_rows h (row := 109) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_8
  simp only [← c4594, ← c4595, ← c4596] at e_109_8
  have e_109_9 := arithEq_of_rows h (row := 109) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_9
  simp only [← c4597, ← c4598, ← c4599] at e_109_9
  have e_110_0 := arithEq_of_rows h (row := 110) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_0
  simp only [← c4600, ← c4601, k3, ← c4602] at e_110_0
  refine ⟨?_, ?_⟩
  · have hc := e_109_8
    simp only [e_108_14] at hc
    linear_combination c4603.trans k2 - hc
  · have hc := e_110_0
    simp only [e_108_13, e_109_9, e_108_14] at hc
    linear_combination c4604.trans k2 - hc

theorem privateBatchWrapper8_f547 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 50)) (a (.virt 61)) (a (.virt 595)) (a (.virt 596)) := by
  have c4605 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4606 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4607 := (privateBatchWrapper8_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4608 := (privateBatchWrapper8_copies144 a h).1
  have c4609 := (privateBatchWrapper8_copies144 a h).2.1
  have c4610 := (privateBatchWrapper8_copies144 a h).2.2.1
  have c4611 := (privateBatchWrapper8_copies144 a h).2.2.2.1
  have c4612 := (privateBatchWrapper8_copies144 a h).2.2.2.2.1
  have c4613 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.1
  have c4614 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.1
  have c4615 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.1
  have c4616 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.1
  have c4617 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.1
  have c4618 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4619 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4620 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4621 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_109_10 := arithEq_of_rows h (row := 109) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_10
  simp only [← c4611, ← c4612, ← c4613] at e_109_10
  have e_109_11 := arithEq_of_rows h (row := 109) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_11
  simp only [← c4614, ← c4615, ← c4616] at e_109_11
  have e_110_1 := arithEq_of_rows h (row := 110) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_1
  simp only [← c4605, k3, ← c4606, k3, ← c4607] at e_110_1
  have e_110_2 := arithEq_of_rows h (row := 110) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_2
  simp only [← c4608, ← c4609, k3, ← c4610] at e_110_2
  have e_110_3 := arithEq_of_rows h (row := 110) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_3
  simp only [← c4617, ← c4618, k3, ← c4619] at e_110_3
  refine ⟨?_, ?_⟩
  · have hc := e_109_10
    simp only [e_110_2] at hc
    linear_combination c4620.trans k2 - hc
  · have hc := e_110_3
    simp only [e_110_1, e_109_11, e_110_2] at hc
    linear_combination c4621.trans k2 - hc

theorem privateBatchWrapper8_f548 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 51) = band (a (.virt 589)) (a (.virt 591)) := by
  have c4622 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4623 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4624 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_12 := arithEq_of_rows h (row := 109) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_12
  simp only [← c4622, ← c4623, ← c4624] at e_109_12
  have hr := e_109_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f549 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 55) = band (a (.virt 593)) (a (.virt 595)) := by
  have c4625 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4626 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4627 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_13 := arithEq_of_rows h (row := 109) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_13
  simp only [← c4625, ← c4626, ← c4627] at e_109_13
  have hr := e_109_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f550 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 109 59) = band (a (.wire 109 51)) (a (.wire 109 55)) := by
  have c4628 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4629 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4630 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_14 := arithEq_of_rows h (row := 109) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_14
  simp only [← c4628, ← c4629, ← c4630] at e_109_14
  have hr := e_109_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f551 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 3) = band (a (.wire 109 15)) (a (.wire 109 59)) := by
  have c4631 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4632 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4633 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_0 := arithEq_of_rows h (row := 111) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_0
  simp only [← c4631, ← c4632, ← c4633] at e_111_0
  have hr := e_111_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f552 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 3) = a (.virt 11) := by
  have c4634 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4634

theorem privateBatchWrapper8_f553 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f554 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 7) = band (a (.wire 17 7)) (a (.wire 18 43)) := by
  have c4635 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4636 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4637 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_1 := arithEq_of_rows h (row := 111) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c4635, ← c4636, ← c4637] at e_111_1
  have hr := e_111_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f555 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 47)) (a (.virt 69)) (a (.virt 597)) (a (.virt 598)) := by
  have c4638 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4639 := (privateBatchWrapper8_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4640 := (privateBatchWrapper8_copies145 a h).1
  have c4641 := (privateBatchWrapper8_copies145 a h).2.1
  have c4642 := (privateBatchWrapper8_copies145 a h).2.2.1
  have c4643 := (privateBatchWrapper8_copies145 a h).2.2.2.1
  have c4644 := (privateBatchWrapper8_copies145 a h).2.2.2.2.1
  have c4645 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.1
  have c4646 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.1
  have c4647 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.1
  have c4648 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.1
  have c4649 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.1
  have c4650 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4651 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4652 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4653 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4654 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_110_4 := arithEq_of_rows h (row := 110) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_4
  simp only [← c4638, k3, ← c4639, k3, ← c4640] at e_110_4
  have e_110_5 := arithEq_of_rows h (row := 110) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_5
  simp only [← c4641, ← c4642, k3, ← c4643] at e_110_5
  have e_110_6 := arithEq_of_rows h (row := 110) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_6
  simp only [← c4650, ← c4651, k3, ← c4652] at e_110_6
  have e_111_2 := arithEq_of_rows h (row := 111) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_2
  simp only [← c4644, ← c4645, ← c4646] at e_111_2
  have e_111_3 := arithEq_of_rows h (row := 111) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_3
  simp only [← c4647, ← c4648, ← c4649] at e_111_3
  refine ⟨?_, ?_⟩
  · have hc := e_111_2
    simp only [e_110_5] at hc
    linear_combination c4653.trans k2 - hc
  · have hc := e_110_6
    simp only [e_110_4, e_111_3, e_110_5] at hc
    linear_combination c4654.trans k2 - hc

theorem privateBatchWrapper8_f556 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 48)) (a (.virt 70)) (a (.virt 599)) (a (.virt 600)) := by
  have c4655 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4656 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4657 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4658 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4659 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4660 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4661 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4662 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4663 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4664 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4665 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4666 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4667 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4668 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4669 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4670 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4671 := (privateBatchWrapper8_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_110_7 := arithEq_of_rows h (row := 110) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_7
  simp only [← c4655, k3, ← c4656, k3, ← c4657] at e_110_7
  have e_110_8 := arithEq_of_rows h (row := 110) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_8
  simp only [← c4658, ← c4659, k3, ← c4660] at e_110_8
  have e_110_9 := arithEq_of_rows h (row := 110) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_9
  simp only [← c4667, ← c4668, k3, ← c4669] at e_110_9
  have e_111_4 := arithEq_of_rows h (row := 111) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_4
  simp only [← c4661, ← c4662, ← c4663] at e_111_4
  have e_111_5 := arithEq_of_rows h (row := 111) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_5
  simp only [← c4664, ← c4665, ← c4666] at e_111_5
  refine ⟨?_, ?_⟩
  · have hc := e_111_4
    simp only [e_110_8] at hc
    linear_combination c4670.trans k2 - hc
  · have hc := e_110_9
    simp only [e_110_7, e_111_5, e_110_8] at hc
    linear_combination c4671.trans k2 - hc

theorem privateBatchWrapper8_f557 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 49)) (a (.virt 71)) (a (.virt 601)) (a (.virt 602)) := by
  have c4672 := (privateBatchWrapper8_copies146 a h).1
  have c4673 := (privateBatchWrapper8_copies146 a h).2.1
  have c4674 := (privateBatchWrapper8_copies146 a h).2.2.1
  have c4675 := (privateBatchWrapper8_copies146 a h).2.2.2.1
  have c4676 := (privateBatchWrapper8_copies146 a h).2.2.2.2.1
  have c4677 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.1
  have c4678 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.1
  have c4679 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.1
  have c4680 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.1
  have c4681 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.1
  have c4682 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4683 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4684 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4685 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4686 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4687 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4688 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_110_10 := arithEq_of_rows h (row := 110) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_10
  simp only [← c4672, k3, ← c4673, k3, ← c4674] at e_110_10
  have e_110_11 := arithEq_of_rows h (row := 110) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_11
  simp only [← c4675, ← c4676, k3, ← c4677] at e_110_11
  have e_110_12 := arithEq_of_rows h (row := 110) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_12
  simp only [← c4684, ← c4685, k3, ← c4686] at e_110_12
  have e_111_6 := arithEq_of_rows h (row := 111) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_6
  simp only [← c4678, ← c4679, ← c4680] at e_111_6
  have e_111_7 := arithEq_of_rows h (row := 111) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_7
  simp only [← c4681, ← c4682, ← c4683] at e_111_7
  refine ⟨?_, ?_⟩
  · have hc := e_111_6
    simp only [e_110_11] at hc
    linear_combination c4687.trans k2 - hc
  · have hc := e_110_12
    simp only [e_110_10, e_111_7, e_110_11] at hc
    linear_combination c4688.trans k2 - hc

theorem privateBatchWrapper8_f558 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 50)) (a (.virt 72)) (a (.virt 603)) (a (.virt 604)) := by
  have c4689 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4690 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4691 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4692 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4693 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4694 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4695 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4696 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4697 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4698 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4699 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4700 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4701 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4702 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4703 := (privateBatchWrapper8_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4704 := (privateBatchWrapper8_copies147 a h).1
  have c4705 := (privateBatchWrapper8_copies147 a h).2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_110_13 := arithEq_of_rows h (row := 110) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_13
  simp only [← c4689, k3, ← c4690, k3, ← c4691] at e_110_13
  have e_110_14 := arithEq_of_rows h (row := 110) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_14
  simp only [← c4692, ← c4693, k3, ← c4694] at e_110_14
  have e_111_8 := arithEq_of_rows h (row := 111) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_8
  simp only [← c4695, ← c4696, ← c4697] at e_111_8
  have e_111_9 := arithEq_of_rows h (row := 111) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_9
  simp only [← c4698, ← c4699, ← c4700] at e_111_9
  have e_112_0 := arithEq_of_rows h (row := 112) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_0
  simp only [← c4701, ← c4702, k3, ← c4703] at e_112_0
  refine ⟨?_, ?_⟩
  · have hc := e_111_8
    simp only [e_110_14] at hc
    linear_combination c4704.trans k2 - hc
  · have hc := e_112_0
    simp only [e_110_13, e_111_9, e_110_14] at hc
    linear_combination c4705.trans k2 - hc

theorem privateBatchWrapper8_f559 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 43) = band (a (.virt 597)) (a (.virt 599)) := by
  have c4706 := (privateBatchWrapper8_copies147 a h).2.2.1
  have c4707 := (privateBatchWrapper8_copies147 a h).2.2.2.1
  have c4708 := (privateBatchWrapper8_copies147 a h).2.2.2.2.1
  have e_111_10 := arithEq_of_rows h (row := 111) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_10
  simp only [← c4706, ← c4707, ← c4708] at e_111_10
  have hr := e_111_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f560 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 47) = band (a (.virt 601)) (a (.virt 603)) := by
  have c4709 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.1
  have c4710 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.1
  have c4711 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.1
  have e_111_11 := arithEq_of_rows h (row := 111) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_11
  simp only [← c4709, ← c4710, ← c4711] at e_111_11
  have hr := e_111_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f561 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 51) = band (a (.wire 111 43)) (a (.wire 111 47)) := by
  have c4712 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.1
  have c4713 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.1
  have c4714 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_111_12 := arithEq_of_rows h (row := 111) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_12
  simp only [← c4712, ← c4713, ← c4714] at e_111_12
  have hr := e_111_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f562 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 55) = band (a (.wire 111 7)) (a (.wire 111 51)) := by
  have c4715 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4716 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4717 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_13 := arithEq_of_rows h (row := 111) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_13
  simp only [← c4715, ← c4716, ← c4717] at e_111_13
  have hr := e_111_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f563 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 55) = a (.virt 11) := by
  have c4718 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4718

theorem privateBatchWrapper8_f564 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f565 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 111 59) = band (a (.wire 17 7)) (a (.wire 20 31)) := by
  have c4719 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4720 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4721 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_14 := arithEq_of_rows h (row := 111) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_14
  simp only [← c4719, ← c4720, ← c4721] at e_111_14
  have hr := e_111_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f566 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 47)) (a (.virt 80)) (a (.virt 605)) (a (.virt 606)) := by
  have c4722 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4723 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4724 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4725 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4726 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4727 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4728 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4729 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4730 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4731 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4732 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4733 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4734 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4735 := (privateBatchWrapper8_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4736 := (privateBatchWrapper8_copies148 a h).1
  have c4737 := (privateBatchWrapper8_copies148 a h).2.1
  have c4738 := (privateBatchWrapper8_copies148 a h).2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_112_1 := arithEq_of_rows h (row := 112) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_1
  simp only [← c4722, k3, ← c4723, k3, ← c4724] at e_112_1
  have e_112_2 := arithEq_of_rows h (row := 112) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_2
  simp only [← c4725, ← c4726, k3, ← c4727] at e_112_2
  have e_112_3 := arithEq_of_rows h (row := 112) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_3
  simp only [← c4734, ← c4735, k3, ← c4736] at e_112_3
  have e_113_0 := arithEq_of_rows h (row := 113) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_0
  simp only [← c4728, ← c4729, ← c4730] at e_113_0
  have e_113_1 := arithEq_of_rows h (row := 113) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_1
  simp only [← c4731, ← c4732, ← c4733] at e_113_1
  refine ⟨?_, ?_⟩
  · have hc := e_113_0
    simp only [e_112_2] at hc
    linear_combination c4737.trans k2 - hc
  · have hc := e_112_3
    simp only [e_112_1, e_113_1, e_112_2] at hc
    linear_combination c4738.trans k2 - hc

theorem privateBatchWrapper8_f567 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 48)) (a (.virt 81)) (a (.virt 607)) (a (.virt 608)) := by
  have c4739 := (privateBatchWrapper8_copies148 a h).2.2.2.1
  have c4740 := (privateBatchWrapper8_copies148 a h).2.2.2.2.1
  have c4741 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.1
  have c4742 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.1
  have c4743 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.1
  have c4744 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.1
  have c4745 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.1
  have c4746 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4747 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4748 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4749 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4750 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4751 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4752 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4753 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4754 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4755 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_112_4 := arithEq_of_rows h (row := 112) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_4
  simp only [← c4739, k3, ← c4740, k3, ← c4741] at e_112_4
  have e_112_5 := arithEq_of_rows h (row := 112) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_5
  simp only [← c4742, ← c4743, k3, ← c4744] at e_112_5
  have e_112_6 := arithEq_of_rows h (row := 112) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_6
  simp only [← c4751, ← c4752, k3, ← c4753] at e_112_6
  have e_113_2 := arithEq_of_rows h (row := 113) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_2
  simp only [← c4745, ← c4746, ← c4747] at e_113_2
  have e_113_3 := arithEq_of_rows h (row := 113) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_3
  simp only [← c4748, ← c4749, ← c4750] at e_113_3
  refine ⟨?_, ?_⟩
  · have hc := e_113_2
    simp only [e_112_5] at hc
    linear_combination c4754.trans k2 - hc
  · have hc := e_112_6
    simp only [e_112_4, e_113_3, e_112_5] at hc
    linear_combination c4755.trans k2 - hc

theorem privateBatchWrapper8_f568 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 49)) (a (.virt 82)) (a (.virt 609)) (a (.virt 610)) := by
  have c4756 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4757 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4758 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4759 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4760 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4761 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4762 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4763 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4764 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4765 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4766 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4767 := (privateBatchWrapper8_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4768 := (privateBatchWrapper8_copies149 a h).1
  have c4769 := (privateBatchWrapper8_copies149 a h).2.1
  have c4770 := (privateBatchWrapper8_copies149 a h).2.2.1
  have c4771 := (privateBatchWrapper8_copies149 a h).2.2.2.1
  have c4772 := (privateBatchWrapper8_copies149 a h).2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_112_7 := arithEq_of_rows h (row := 112) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_7
  simp only [← c4756, k3, ← c4757, k3, ← c4758] at e_112_7
  have e_112_8 := arithEq_of_rows h (row := 112) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_8
  simp only [← c4759, ← c4760, k3, ← c4761] at e_112_8
  have e_112_9 := arithEq_of_rows h (row := 112) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_9
  simp only [← c4768, ← c4769, k3, ← c4770] at e_112_9
  have e_113_4 := arithEq_of_rows h (row := 113) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_4
  simp only [← c4762, ← c4763, ← c4764] at e_113_4
  have e_113_5 := arithEq_of_rows h (row := 113) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_5
  simp only [← c4765, ← c4766, ← c4767] at e_113_5
  refine ⟨?_, ?_⟩
  · have hc := e_113_4
    simp only [e_112_8] at hc
    linear_combination c4771.trans k2 - hc
  · have hc := e_112_9
    simp only [e_112_7, e_113_5, e_112_8] at hc
    linear_combination c4772.trans k2 - hc

theorem privateBatchWrapper8_f569 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 50)) (a (.virt 83)) (a (.virt 611)) (a (.virt 612)) := by
  have c4773 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.1
  have c4774 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.1
  have c4775 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.1
  have c4776 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.1
  have c4777 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.1
  have c4778 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4779 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4780 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4781 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4782 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4783 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4784 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4785 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4786 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4787 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4788 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4789 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_112_10 := arithEq_of_rows h (row := 112) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_10
  simp only [← c4773, k3, ← c4774, k3, ← c4775] at e_112_10
  have e_112_11 := arithEq_of_rows h (row := 112) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_11
  simp only [← c4776, ← c4777, k3, ← c4778] at e_112_11
  have e_112_12 := arithEq_of_rows h (row := 112) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_12
  simp only [← c4785, ← c4786, k3, ← c4787] at e_112_12
  have e_113_6 := arithEq_of_rows h (row := 113) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_6
  simp only [← c4779, ← c4780, ← c4781] at e_113_6
  have e_113_7 := arithEq_of_rows h (row := 113) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_7
  simp only [← c4782, ← c4783, ← c4784] at e_113_7
  refine ⟨?_, ?_⟩
  · have hc := e_113_6
    simp only [e_112_11] at hc
    linear_combination c4788.trans k2 - hc
  · have hc := e_112_12
    simp only [e_112_10, e_113_7, e_112_11] at hc
    linear_combination c4789.trans k2 - hc

theorem privateBatchWrapper8_f570 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 113 35) = band (a (.virt 605)) (a (.virt 607)) := by
  have c4790 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4791 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4792 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_8 := arithEq_of_rows h (row := 113) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_8
  simp only [← c4790, ← c4791, ← c4792] at e_113_8
  have hr := e_113_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f571 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 113 39) = band (a (.virt 609)) (a (.virt 611)) := by
  have c4793 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4794 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4795 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_9 := arithEq_of_rows h (row := 113) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_9
  simp only [← c4793, ← c4794, ← c4795] at e_113_9
  have hr := e_113_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f572 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 113 43) = band (a (.wire 113 35)) (a (.wire 113 39)) := by
  have c4796 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4797 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4798 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_10 := arithEq_of_rows h (row := 113) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_10
  simp only [← c4796, ← c4797, ← c4798] at e_113_10
  have hr := e_113_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f573 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 113 47) = band (a (.wire 111 59)) (a (.wire 113 43)) := by
  have c4799 := (privateBatchWrapper8_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4800 := (privateBatchWrapper8_copies150 a h).1
  have c4801 := (privateBatchWrapper8_copies150 a h).2.1
  have e_113_11 := arithEq_of_rows h (row := 113) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_11
  simp only [← c4799, ← c4800, ← c4801] at e_113_11
  have hr := e_113_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f574 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 113 47) = a (.virt 11) := by
  have c4802 := (privateBatchWrapper8_copies150 a h).2.2.1
  exact c4802

theorem privateBatchWrapper8_f575 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 17 55) = bnot (a (.wire 10 11)) := by
  have c718 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper8_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c718, k3, ← c719, k3, ← c720] at e_17_13
  have hr := e_17_13
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f576 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f577 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 113 51) = band (a (.wire 17 55)) (a (.wire 18 43)) := by
  have c4803 := (privateBatchWrapper8_copies150 a h).2.2.2.1
  have c4804 := (privateBatchWrapper8_copies150 a h).2.2.2.2.1
  have c4805 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.1
  have e_113_12 := arithEq_of_rows h (row := 113) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_12
  simp only [← c4803, ← c4804, ← c4805] at e_113_12
  have hr := e_113_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f578 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 58)) (a (.virt 69)) (a (.virt 613)) (a (.virt 614)) := by
  have c4806 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.1
  have c4807 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.1
  have c4808 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.1
  have c4809 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.1
  have c4810 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4811 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4812 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4813 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4814 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4815 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4816 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4817 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4818 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4819 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4820 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4821 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4822 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_112_13 := arithEq_of_rows h (row := 112) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_13
  simp only [← c4806, k3, ← c4807, k3, ← c4808] at e_112_13
  have e_112_14 := arithEq_of_rows h (row := 112) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_14
  simp only [← c4809, ← c4810, k3, ← c4811] at e_112_14
  have e_113_13 := arithEq_of_rows h (row := 113) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_13
  simp only [← c4812, ← c4813, ← c4814] at e_113_13
  have e_113_14 := arithEq_of_rows h (row := 113) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_14
  simp only [← c4815, ← c4816, ← c4817] at e_113_14
  have e_114_0 := arithEq_of_rows h (row := 114) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_0
  simp only [← c4818, ← c4819, k3, ← c4820] at e_114_0
  refine ⟨?_, ?_⟩
  · have hc := e_113_13
    simp only [e_112_14] at hc
    linear_combination c4821.trans k2 - hc
  · have hc := e_114_0
    simp only [e_112_13, e_113_14, e_112_14] at hc
    linear_combination c4822.trans k2 - hc

theorem privateBatchWrapper8_f579 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 59)) (a (.virt 70)) (a (.virt 615)) (a (.virt 616)) := by
  have c4823 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4824 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4825 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4826 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4827 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4828 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4829 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4830 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4831 := (privateBatchWrapper8_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4832 := (privateBatchWrapper8_copies151 a h).1
  have c4833 := (privateBatchWrapper8_copies151 a h).2.1
  have c4834 := (privateBatchWrapper8_copies151 a h).2.2.1
  have c4835 := (privateBatchWrapper8_copies151 a h).2.2.2.1
  have c4836 := (privateBatchWrapper8_copies151 a h).2.2.2.2.1
  have c4837 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.1
  have c4838 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.1
  have c4839 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_114_1 := arithEq_of_rows h (row := 114) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_1
  simp only [← c4823, k3, ← c4824, k3, ← c4825] at e_114_1
  have e_114_2 := arithEq_of_rows h (row := 114) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_2
  simp only [← c4826, ← c4827, k3, ← c4828] at e_114_2
  have e_114_3 := arithEq_of_rows h (row := 114) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_3
  simp only [← c4835, ← c4836, k3, ← c4837] at e_114_3
  have e_115_0 := arithEq_of_rows h (row := 115) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_0
  simp only [← c4829, ← c4830, ← c4831] at e_115_0
  have e_115_1 := arithEq_of_rows h (row := 115) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_1
  simp only [← c4832, ← c4833, ← c4834] at e_115_1
  refine ⟨?_, ?_⟩
  · have hc := e_115_0
    simp only [e_114_2] at hc
    linear_combination c4838.trans k2 - hc
  · have hc := e_114_3
    simp only [e_114_1, e_115_1, e_114_2] at hc
    linear_combination c4839.trans k2 - hc

theorem privateBatchWrapper8_f580 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 60)) (a (.virt 71)) (a (.virt 617)) (a (.virt 618)) := by
  have c4840 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.1
  have c4841 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.1
  have c4842 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4843 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4844 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4845 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4846 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4847 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4848 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4849 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4850 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4851 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4852 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4853 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4854 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4855 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4856 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_114_4 := arithEq_of_rows h (row := 114) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_4
  simp only [← c4840, k3, ← c4841, k3, ← c4842] at e_114_4
  have e_114_5 := arithEq_of_rows h (row := 114) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_5
  simp only [← c4843, ← c4844, k3, ← c4845] at e_114_5
  have e_114_6 := arithEq_of_rows h (row := 114) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_6
  simp only [← c4852, ← c4853, k3, ← c4854] at e_114_6
  have e_115_2 := arithEq_of_rows h (row := 115) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_2
  simp only [← c4846, ← c4847, ← c4848] at e_115_2
  have e_115_3 := arithEq_of_rows h (row := 115) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_3
  simp only [← c4849, ← c4850, ← c4851] at e_115_3
  refine ⟨?_, ?_⟩
  · have hc := e_115_2
    simp only [e_114_5] at hc
    linear_combination c4855.trans k2 - hc
  · have hc := e_114_6
    simp only [e_114_4, e_115_3, e_114_5] at hc
    linear_combination c4856.trans k2 - hc

theorem privateBatchWrapper8_f581 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 61)) (a (.virt 72)) (a (.virt 619)) (a (.virt 620)) := by
  have c4857 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4858 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4859 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4860 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4861 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4862 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4863 := (privateBatchWrapper8_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4864 := (privateBatchWrapper8_copies152 a h).1
  have c4865 := (privateBatchWrapper8_copies152 a h).2.1
  have c4866 := (privateBatchWrapper8_copies152 a h).2.2.1
  have c4867 := (privateBatchWrapper8_copies152 a h).2.2.2.1
  have c4868 := (privateBatchWrapper8_copies152 a h).2.2.2.2.1
  have c4869 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.1
  have c4870 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.1
  have c4871 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.1
  have c4872 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.1
  have c4873 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_114_7 := arithEq_of_rows h (row := 114) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_7
  simp only [← c4857, k3, ← c4858, k3, ← c4859] at e_114_7
  have e_114_8 := arithEq_of_rows h (row := 114) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_8
  simp only [← c4860, ← c4861, k3, ← c4862] at e_114_8
  have e_114_9 := arithEq_of_rows h (row := 114) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_9
  simp only [← c4869, ← c4870, k3, ← c4871] at e_114_9
  have e_115_4 := arithEq_of_rows h (row := 115) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_4
  simp only [← c4863, ← c4864, ← c4865] at e_115_4
  have e_115_5 := arithEq_of_rows h (row := 115) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_5
  simp only [← c4866, ← c4867, ← c4868] at e_115_5
  refine ⟨?_, ?_⟩
  · have hc := e_115_4
    simp only [e_114_8] at hc
    linear_combination c4872.trans k2 - hc
  · have hc := e_114_9
    simp only [e_114_7, e_115_5, e_114_8] at hc
    linear_combination c4873.trans k2 - hc

theorem privateBatchWrapper8_f582 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 115 27) = band (a (.virt 613)) (a (.virt 615)) := by
  have c4874 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4875 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4876 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_6 := arithEq_of_rows h (row := 115) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_6
  simp only [← c4874, ← c4875, ← c4876] at e_115_6
  have hr := e_115_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f583 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 115 31) = band (a (.virt 617)) (a (.virt 619)) := by
  have c4877 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4878 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4879 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_7 := arithEq_of_rows h (row := 115) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_7
  simp only [← c4877, ← c4878, ← c4879] at e_115_7
  have hr := e_115_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f584 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 115 35) = band (a (.wire 115 27)) (a (.wire 115 31)) := by
  have c4880 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4881 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4882 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_8 := arithEq_of_rows h (row := 115) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_8
  simp only [← c4880, ← c4881, ← c4882] at e_115_8
  have hr := e_115_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f585 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 115 39) = band (a (.wire 113 51)) (a (.wire 115 35)) := by
  have c4883 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4884 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4885 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_9 := arithEq_of_rows h (row := 115) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_9
  simp only [← c4883, ← c4884, ← c4885] at e_115_9
  have hr := e_115_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f586 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 115 39) = a (.virt 11) := by
  have c4886 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c4886

theorem privateBatchWrapper8_f587 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f588 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 115 43) = band (a (.wire 17 55)) (a (.wire 20 31)) := by
  have c4887 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4888 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4889 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_10 := arithEq_of_rows h (row := 115) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_10
  simp only [← c4887, ← c4888, ← c4889] at e_115_10
  have hr := e_115_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f589 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 58)) (a (.virt 80)) (a (.virt 621)) (a (.virt 622)) := by
  have c4890 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4891 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4892 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4893 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4894 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4895 := (privateBatchWrapper8_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4896 := (privateBatchWrapper8_copies153 a h).1
  have c4897 := (privateBatchWrapper8_copies153 a h).2.1
  have c4898 := (privateBatchWrapper8_copies153 a h).2.2.1
  have c4899 := (privateBatchWrapper8_copies153 a h).2.2.2.1
  have c4900 := (privateBatchWrapper8_copies153 a h).2.2.2.2.1
  have c4901 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.1
  have c4902 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.1
  have c4903 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.1
  have c4904 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.1
  have c4905 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.1
  have c4906 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_114_10 := arithEq_of_rows h (row := 114) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_10
  simp only [← c4890, k3, ← c4891, k3, ← c4892] at e_114_10
  have e_114_11 := arithEq_of_rows h (row := 114) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_11
  simp only [← c4893, ← c4894, k3, ← c4895] at e_114_11
  have e_114_12 := arithEq_of_rows h (row := 114) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_12
  simp only [← c4902, ← c4903, k3, ← c4904] at e_114_12
  have e_115_11 := arithEq_of_rows h (row := 115) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_11
  simp only [← c4896, ← c4897, ← c4898] at e_115_11
  have e_115_12 := arithEq_of_rows h (row := 115) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_12
  simp only [← c4899, ← c4900, ← c4901] at e_115_12
  refine ⟨?_, ?_⟩
  · have hc := e_115_11
    simp only [e_114_11] at hc
    linear_combination c4905.trans k2 - hc
  · have hc := e_114_12
    simp only [e_114_10, e_115_12, e_114_11] at hc
    linear_combination c4906.trans k2 - hc

theorem privateBatchWrapper8_f590 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 59)) (a (.virt 81)) (a (.virt 623)) (a (.virt 624)) := by
  have c4907 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4908 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4910 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4913 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4914 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4915 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4916 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4917 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4918 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4919 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4920 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4921 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4922 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4923 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_114_13 := arithEq_of_rows h (row := 114) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_13
  simp only [← c4907, k3, ← c4908, k3, ← c4909] at e_114_13
  have e_114_14 := arithEq_of_rows h (row := 114) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_14
  simp only [← c4910, ← c4911, k3, ← c4912] at e_114_14
  have e_115_13 := arithEq_of_rows h (row := 115) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_13
  simp only [← c4913, ← c4914, ← c4915] at e_115_13
  have e_115_14 := arithEq_of_rows h (row := 115) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_14
  simp only [← c4916, ← c4917, ← c4918] at e_115_14
  have e_116_0 := arithEq_of_rows h (row := 116) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_0
  simp only [← c4919, ← c4920, k3, ← c4921] at e_116_0
  refine ⟨?_, ?_⟩
  · have hc := e_115_13
    simp only [e_114_14] at hc
    linear_combination c4922.trans k2 - hc
  · have hc := e_116_0
    simp only [e_114_13, e_115_14, e_114_14] at hc
    linear_combination c4923.trans k2 - hc

theorem privateBatchWrapper8_f591 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 60)) (a (.virt 82)) (a (.virt 625)) (a (.virt 626)) := by
  have c4924 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4925 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4926 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4927 := (privateBatchWrapper8_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4928 := (privateBatchWrapper8_copies154 a h).1
  have c4929 := (privateBatchWrapper8_copies154 a h).2.1
  have c4930 := (privateBatchWrapper8_copies154 a h).2.2.1
  have c4931 := (privateBatchWrapper8_copies154 a h).2.2.2.1
  have c4932 := (privateBatchWrapper8_copies154 a h).2.2.2.2.1
  have c4933 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.1
  have c4934 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.1
  have c4935 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.1
  have c4936 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.1
  have c4937 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.1
  have c4938 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4939 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4940 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_116_1 := arithEq_of_rows h (row := 116) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_1
  simp only [← c4924, k3, ← c4925, k3, ← c4926] at e_116_1
  have e_116_2 := arithEq_of_rows h (row := 116) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_2
  simp only [← c4927, ← c4928, k3, ← c4929] at e_116_2
  have e_116_3 := arithEq_of_rows h (row := 116) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_3
  simp only [← c4936, ← c4937, k3, ← c4938] at e_116_3
  have e_117_0 := arithEq_of_rows h (row := 117) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_0
  simp only [← c4930, ← c4931, ← c4932] at e_117_0
  have e_117_1 := arithEq_of_rows h (row := 117) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_1
  simp only [← c4933, ← c4934, ← c4935] at e_117_1
  refine ⟨?_, ?_⟩
  · have hc := e_117_0
    simp only [e_116_2] at hc
    linear_combination c4939.trans k2 - hc
  · have hc := e_116_3
    simp only [e_116_1, e_117_1, e_116_2] at hc
    linear_combination c4940.trans k2 - hc

theorem privateBatchWrapper8_f592 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 61)) (a (.virt 83)) (a (.virt 627)) (a (.virt 628)) := by
  have c4941 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4942 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4943 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4944 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4945 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4946 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4947 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4948 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4949 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4950 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4951 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4952 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4953 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4954 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4955 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4956 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4957 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_116_4 := arithEq_of_rows h (row := 116) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_4
  simp only [← c4941, k3, ← c4942, k3, ← c4943] at e_116_4
  have e_116_5 := arithEq_of_rows h (row := 116) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_5
  simp only [← c4944, ← c4945, k3, ← c4946] at e_116_5
  have e_116_6 := arithEq_of_rows h (row := 116) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_6
  simp only [← c4953, ← c4954, k3, ← c4955] at e_116_6
  have e_117_2 := arithEq_of_rows h (row := 117) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_2
  simp only [← c4947, ← c4948, ← c4949] at e_117_2
  have e_117_3 := arithEq_of_rows h (row := 117) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_3
  simp only [← c4950, ← c4951, ← c4952] at e_117_3
  refine ⟨?_, ?_⟩
  · have hc := e_117_2
    simp only [e_116_5] at hc
    linear_combination c4956.trans k2 - hc
  · have hc := e_116_6
    simp only [e_116_4, e_117_3, e_116_5] at hc
    linear_combination c4957.trans k2 - hc

theorem privateBatchWrapper8_f593 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 117 19) = band (a (.virt 621)) (a (.virt 623)) := by
  have c4958 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4959 := (privateBatchWrapper8_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4960 := (privateBatchWrapper8_copies155 a h).1
  have e_117_4 := arithEq_of_rows h (row := 117) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_4
  simp only [← c4958, ← c4959, ← c4960] at e_117_4
  have hr := e_117_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f594 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 117 23) = band (a (.virt 625)) (a (.virt 627)) := by
  have c4961 := (privateBatchWrapper8_copies155 a h).2.1
  have c4962 := (privateBatchWrapper8_copies155 a h).2.2.1
  have c4963 := (privateBatchWrapper8_copies155 a h).2.2.2.1
  have e_117_5 := arithEq_of_rows h (row := 117) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_5
  simp only [← c4961, ← c4962, ← c4963] at e_117_5
  have hr := e_117_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f595 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 117 27) = band (a (.wire 117 19)) (a (.wire 117 23)) := by
  have c4964 := (privateBatchWrapper8_copies155 a h).2.2.2.2.1
  have c4965 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.1
  have c4966 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.1
  have e_117_6 := arithEq_of_rows h (row := 117) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_6
  simp only [← c4964, ← c4965, ← c4966] at e_117_6
  have hr := e_117_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f596 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 117 31) = band (a (.wire 115 43)) (a (.wire 117 27)) := by
  have c4967 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.1
  have c4968 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.1
  have c4969 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.1
  have e_117_7 := arithEq_of_rows h (row := 117) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_7
  simp only [← c4967, ← c4968, ← c4969] at e_117_7
  have hr := e_117_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f597 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 117 31) = a (.virt 11) := by
  have c4970 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c4970

theorem privateBatchWrapper8_f598 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 18 43) = bnot (a (.wire 10 55)) := by
  have c763 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper8_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c763, k3, ← c764, k3, ← c765] at e_18_10
  have hr := e_18_10
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f599 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f600 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 117 35) = band (a (.wire 18 43)) (a (.wire 20 31)) := by
  have c4971 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4972 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4973 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_8 := arithEq_of_rows h (row := 117) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_8
  simp only [← c4971, ← c4972, ← c4973] at e_117_8
  have hr := e_117_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f601 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 69)) (a (.virt 80)) (a (.virt 629)) (a (.virt 630)) := by
  have c4974 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4975 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4976 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4977 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4978 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4979 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4980 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4981 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4982 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4983 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4984 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4985 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4986 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4987 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4988 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4989 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4990 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_116_7 := arithEq_of_rows h (row := 116) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_7
  simp only [← c4974, k3, ← c4975, k3, ← c4976] at e_116_7
  have e_116_8 := arithEq_of_rows h (row := 116) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_8
  simp only [← c4977, ← c4978, k3, ← c4979] at e_116_8
  have e_116_9 := arithEq_of_rows h (row := 116) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_9
  simp only [← c4986, ← c4987, k3, ← c4988] at e_116_9
  have e_117_9 := arithEq_of_rows h (row := 117) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_9
  simp only [← c4980, ← c4981, ← c4982] at e_117_9
  have e_117_10 := arithEq_of_rows h (row := 117) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_10
  simp only [← c4983, ← c4984, ← c4985] at e_117_10
  refine ⟨?_, ?_⟩
  · have hc := e_117_9
    simp only [e_116_8] at hc
    linear_combination c4989.trans k2 - hc
  · have hc := e_116_9
    simp only [e_116_7, e_117_10, e_116_8] at hc
    linear_combination c4990.trans k2 - hc

theorem privateBatchWrapper8_f602 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 70)) (a (.virt 81)) (a (.virt 631)) (a (.virt 632)) := by
  have c4991 := (privateBatchWrapper8_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4992 := (privateBatchWrapper8_copies156 a h).1
  have c4993 := (privateBatchWrapper8_copies156 a h).2.1
  have c4994 := (privateBatchWrapper8_copies156 a h).2.2.1
  have c4995 := (privateBatchWrapper8_copies156 a h).2.2.2.1
  have c4996 := (privateBatchWrapper8_copies156 a h).2.2.2.2.1
  have c4997 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.1
  have c4998 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.1
  have c4999 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.1
  have c5000 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.1
  have c5001 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.1
  have c5002 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5003 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5004 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5005 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5006 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5007 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_116_10 := arithEq_of_rows h (row := 116) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_10
  simp only [← c4991, k3, ← c4992, k3, ← c4993] at e_116_10
  have e_116_11 := arithEq_of_rows h (row := 116) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_11
  simp only [← c4994, ← c4995, k3, ← c4996] at e_116_11
  have e_116_12 := arithEq_of_rows h (row := 116) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_12
  simp only [← c5003, ← c5004, k3, ← c5005] at e_116_12
  have e_117_11 := arithEq_of_rows h (row := 117) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_11
  simp only [← c4997, ← c4998, ← c4999] at e_117_11
  have e_117_12 := arithEq_of_rows h (row := 117) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_12
  simp only [← c5000, ← c5001, ← c5002] at e_117_12
  refine ⟨?_, ?_⟩
  · have hc := e_117_11
    simp only [e_116_11] at hc
    linear_combination c5006.trans k2 - hc
  · have hc := e_116_12
    simp only [e_116_10, e_117_12, e_116_11] at hc
    linear_combination c5007.trans k2 - hc

theorem privateBatchWrapper8_f603 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 71)) (a (.virt 82)) (a (.virt 633)) (a (.virt 634)) := by
  have c5008 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5009 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5010 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5011 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5012 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5013 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5014 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5015 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5016 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5017 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5018 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5019 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5020 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5021 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5022 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5023 := (privateBatchWrapper8_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5024 := (privateBatchWrapper8_copies157 a h).1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_116_13 := arithEq_of_rows h (row := 116) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_13
  simp only [← c5008, k3, ← c5009, k3, ← c5010] at e_116_13
  have e_116_14 := arithEq_of_rows h (row := 116) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_14
  simp only [← c5011, ← c5012, k3, ← c5013] at e_116_14
  have e_117_13 := arithEq_of_rows h (row := 117) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_13
  simp only [← c5014, ← c5015, ← c5016] at e_117_13
  have e_117_14 := arithEq_of_rows h (row := 117) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_14
  simp only [← c5017, ← c5018, ← c5019] at e_117_14
  have e_118_0 := arithEq_of_rows h (row := 118) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_0
  simp only [← c5020, ← c5021, k3, ← c5022] at e_118_0
  refine ⟨?_, ?_⟩
  · have hc := e_117_13
    simp only [e_116_14] at hc
    linear_combination c5023.trans k2 - hc
  · have hc := e_118_0
    simp only [e_116_13, e_117_14, e_116_14] at hc
    linear_combination c5024.trans k2 - hc

theorem privateBatchWrapper8_f604 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    IsEqual (a (.virt 72)) (a (.virt 83)) (a (.virt 635)) (a (.virt 636)) := by
  have c5025 := (privateBatchWrapper8_copies157 a h).2.1
  have c5026 := (privateBatchWrapper8_copies157 a h).2.2.1
  have c5027 := (privateBatchWrapper8_copies157 a h).2.2.2.1
  have c5028 := (privateBatchWrapper8_copies157 a h).2.2.2.2.1
  have c5029 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.1
  have c5030 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.1
  have c5031 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.1
  have c5032 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.1
  have c5033 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.1
  have c5034 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5035 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5036 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5037 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5038 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5039 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5040 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5041 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_118_1 := arithEq_of_rows h (row := 118) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_1
  simp only [← c5025, k3, ← c5026, k3, ← c5027] at e_118_1
  have e_118_2 := arithEq_of_rows h (row := 118) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_2
  simp only [← c5028, ← c5029, k3, ← c5030] at e_118_2
  have e_118_3 := arithEq_of_rows h (row := 118) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_3
  simp only [← c5037, ← c5038, k3, ← c5039] at e_118_3
  have e_119_0 := arithEq_of_rows h (row := 119) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_0
  simp only [← c5031, ← c5032, ← c5033] at e_119_0
  have e_119_1 := arithEq_of_rows h (row := 119) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_1
  simp only [← c5034, ← c5035, ← c5036] at e_119_1
  refine ⟨?_, ?_⟩
  · have hc := e_119_0
    simp only [e_118_2] at hc
    linear_combination c5040.trans k2 - hc
  · have hc := e_118_3
    simp only [e_118_1, e_119_1, e_118_2] at hc
    linear_combination c5041.trans k2 - hc

theorem privateBatchWrapper8_f605 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 119 11) = band (a (.virt 629)) (a (.virt 631)) := by
  have c5042 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5043 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5044 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_2 := arithEq_of_rows h (row := 119) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_2
  simp only [← c5042, ← c5043, ← c5044] at e_119_2
  have hr := e_119_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f606 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 119 15) = band (a (.virt 633)) (a (.virt 635)) := by
  have c5045 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5046 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5047 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_3 := arithEq_of_rows h (row := 119) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_3
  simp only [← c5045, ← c5046, ← c5047] at e_119_3
  have hr := e_119_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f607 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 119 19) = band (a (.wire 119 11)) (a (.wire 119 15)) := by
  have c5048 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5049 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5050 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_4 := arithEq_of_rows h (row := 119) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_4
  simp only [← c5048, ← c5049, ← c5050] at e_119_4
  have hr := e_119_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f608 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 119 23) = band (a (.wire 117 35)) (a (.wire 119 19)) := by
  have c5051 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5052 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5053 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_5 := arithEq_of_rows h (row := 119) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_5
  simp only [← c5051, ← c5052, ← c5053] at e_119_5
  have hr := e_119_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper8_f609 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 119 23) = a (.virt 11) := by
  have c5054 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c5054

theorem privateBatchWrapper8_f610 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 20 31) = bnot (a (.wire 11 39)) := by
  have c808 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper8_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c808, k3, ← c809, k3, ← c810] at e_20_7
  have hr := e_20_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper8_f611 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 120 12) = spongeHash perm [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254)] 0 ∧ a (.wire 120 13) = spongeHash perm [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254)] 1 ∧ a (.wire 120 14) = spongeHash perm [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254)] 2 ∧ a (.wire 120 15) = spongeHash perm [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254)] 3) := by
  have c5055 := (privateBatchWrapper8_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5056 := (privateBatchWrapper8_copies158 a h).1
  have c5057 := (privateBatchWrapper8_copies158 a h).2.1
  have c5058 := (privateBatchWrapper8_copies158 a h).2.2.1
  have c5059 := (privateBatchWrapper8_copies158 a h).2.2.2.1
  have c5060 := (privateBatchWrapper8_copies158 a h).2.2.2.2.1
  have c5061 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.1
  have c5062 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.1
  have c5063 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.1
  have c5064 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.1
  have c5065 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.1
  have c5066 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 120 = addBlock (fun _ => 0) [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), 1, 0, 0, 0] :=
    poseidon2In_first c5055.symm c5056.symm c5057.symm c5058.symm (c5059.symm.trans k3) (c5060.symm.trans k2) (c5061.symm.trans k2) (c5062.symm.trans k2) (c5063.symm.trans k2) (c5064.symm.trans k2) (c5065.symm.trans k2) (c5066.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 120) rfl rfl hin0
  have hpad : pad10 [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254)] =
      [a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f612 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 121 12) = spongeHash perm [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15)] 0 ∧ a (.wire 121 13) = spongeHash perm [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15)] 1 ∧ a (.wire 121 14) = spongeHash perm [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15)] 2 ∧ a (.wire 121 15) = spongeHash perm [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15)] 3) := by
  have c5067 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5068 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5069 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5070 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5071 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5072 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5073 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5074 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5075 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5076 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5077 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5078 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 121 = addBlock (fun _ => 0) [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15), 1, 0, 0, 0] :=
    poseidon2In_first c5067.symm c5068.symm c5069.symm c5070.symm (c5071.symm.trans k3) (c5072.symm.trans k2) (c5073.symm.trans k2) (c5074.symm.trans k2) (c5075.symm.trans k2) (c5076.symm.trans k2) (c5077.symm.trans k2) (c5078.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 121) rfl rfl hin0
  have hpad : pad10 [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15)] =
      [a (.wire 120 12), a (.wire 120 13), a (.wire 120 14), a (.wire 120 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f613 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 118 23) = bselect (a (.wire 3 31)) (a (.wire 121 12)) (a (.virt 1)) := by
  have c5079 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5080 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5081 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5082 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5083 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5084 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_4 := arithEq_of_rows h (row := 118) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_4
  simp only [← c5079, ← c5080, ← c5081] at e_118_4
  have e_118_5 := arithEq_of_rows h (row := 118) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_5
  simp only [← c5082, ← c5083, ← c5084] at e_118_5
  have hr := e_118_5
  simp only [e_118_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f614 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 118 31) = bselect (a (.wire 3 31)) (a (.wire 121 13)) (a (.virt 2)) := by
  have c5085 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5086 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5087 := (privateBatchWrapper8_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5088 := (privateBatchWrapper8_copies159 a h).1
  have c5089 := (privateBatchWrapper8_copies159 a h).2.1
  have c5090 := (privateBatchWrapper8_copies159 a h).2.2.1
  have e_118_6 := arithEq_of_rows h (row := 118) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_6
  simp only [← c5085, ← c5086, ← c5087] at e_118_6
  have e_118_7 := arithEq_of_rows h (row := 118) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_7
  simp only [← c5088, ← c5089, ← c5090] at e_118_7
  have hr := e_118_7
  simp only [e_118_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f615 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 118 39) = bselect (a (.wire 3 31)) (a (.wire 121 14)) (a (.virt 3)) := by
  have c5091 := (privateBatchWrapper8_copies159 a h).2.2.2.1
  have c5092 := (privateBatchWrapper8_copies159 a h).2.2.2.2.1
  have c5093 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.1
  have c5094 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.1
  have c5095 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.1
  have c5096 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.1
  have e_118_8 := arithEq_of_rows h (row := 118) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_8
  simp only [← c5091, ← c5092, ← c5093] at e_118_8
  have e_118_9 := arithEq_of_rows h (row := 118) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_9
  simp only [← c5094, ← c5095, ← c5096] at e_118_9
  have hr := e_118_9
  simp only [e_118_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f616 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 118 47) = bselect (a (.wire 3 31)) (a (.wire 121 15)) (a (.virt 4)) := by
  have c5097 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.1
  have c5098 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5099 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5100 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5101 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5102 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_10 := arithEq_of_rows h (row := 118) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_10
  simp only [← c5097, ← c5098, ← c5099] at e_118_10
  have e_118_11 := arithEq_of_rows h (row := 118) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_11
  simp only [← c5100, ← c5101, ← c5102] at e_118_11
  have hr := e_118_11
  simp only [e_118_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f617 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 122 12) = spongeHash perm [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258)] 0 ∧ a (.wire 122 13) = spongeHash perm [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258)] 1 ∧ a (.wire 122 14) = spongeHash perm [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258)] 2 ∧ a (.wire 122 15) = spongeHash perm [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258)] 3) := by
  have c5103 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5104 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5105 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5106 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5107 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5108 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5109 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5110 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5111 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5112 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5113 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5114 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 122 = addBlock (fun _ => 0) [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), 1, 0, 0, 0] :=
    poseidon2In_first c5103.symm c5104.symm c5105.symm c5106.symm (c5107.symm.trans k3) (c5108.symm.trans k2) (c5109.symm.trans k2) (c5110.symm.trans k2) (c5111.symm.trans k2) (c5112.symm.trans k2) (c5113.symm.trans k2) (c5114.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 122) rfl rfl hin0
  have hpad : pad10 [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258)] =
      [a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f618 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 123 12) = spongeHash perm [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15)] 0 ∧ a (.wire 123 13) = spongeHash perm [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15)] 1 ∧ a (.wire 123 14) = spongeHash perm [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15)] 2 ∧ a (.wire 123 15) = spongeHash perm [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15)] 3) := by
  have c5115 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5116 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5117 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5118 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5119 := (privateBatchWrapper8_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5120 := (privateBatchWrapper8_copies160 a h).1
  have c5121 := (privateBatchWrapper8_copies160 a h).2.1
  have c5122 := (privateBatchWrapper8_copies160 a h).2.2.1
  have c5123 := (privateBatchWrapper8_copies160 a h).2.2.2.1
  have c5124 := (privateBatchWrapper8_copies160 a h).2.2.2.2.1
  have c5125 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.1
  have c5126 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 123 = addBlock (fun _ => 0) [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15), 1, 0, 0, 0] :=
    poseidon2In_first c5115.symm c5116.symm c5117.symm c5118.symm (c5119.symm.trans k3) (c5120.symm.trans k2) (c5121.symm.trans k2) (c5122.symm.trans k2) (c5123.symm.trans k2) (c5124.symm.trans k2) (c5125.symm.trans k2) (c5126.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 123) rfl rfl hin0
  have hpad : pad10 [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15)] =
      [a (.wire 122 12), a (.wire 122 13), a (.wire 122 14), a (.wire 122 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f619 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 118 55) = bselect (a (.wire 4 15)) (a (.wire 123 12)) (a (.virt 14)) := by
  have c5127 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.1
  have c5128 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.1
  have c5129 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.1
  have c5130 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5131 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5132 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_12 := arithEq_of_rows h (row := 118) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_12
  simp only [← c5127, ← c5128, ← c5129] at e_118_12
  have e_118_13 := arithEq_of_rows h (row := 118) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_13
  simp only [← c5130, ← c5131, ← c5132] at e_118_13
  have hr := e_118_13
  simp only [e_118_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f620 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 3) = bselect (a (.wire 4 15)) (a (.wire 123 13)) (a (.virt 15)) := by
  have c5133 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5134 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5135 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5136 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5137 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5138 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_14 := arithEq_of_rows h (row := 118) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_14
  simp only [← c5133, ← c5134, ← c5135] at e_118_14
  have e_124_0 := arithEq_of_rows h (row := 124) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_0
  simp only [← c5136, ← c5137, ← c5138] at e_124_0
  have hr := e_124_0
  simp only [e_118_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f621 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 11) = bselect (a (.wire 4 15)) (a (.wire 123 14)) (a (.virt 16)) := by
  have c5139 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5140 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5141 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5142 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5143 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5144 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_1 := arithEq_of_rows h (row := 124) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_1
  simp only [← c5139, ← c5140, ← c5141] at e_124_1
  have e_124_2 := arithEq_of_rows h (row := 124) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_2
  simp only [← c5142, ← c5143, ← c5144] at e_124_2
  have hr := e_124_2
  simp only [e_124_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f622 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 19) = bselect (a (.wire 4 15)) (a (.wire 123 15)) (a (.virt 17)) := by
  have c5145 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5146 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5147 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5148 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5149 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5150 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_3 := arithEq_of_rows h (row := 124) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_3
  simp only [← c5145, ← c5146, ← c5147] at e_124_3
  have e_124_4 := arithEq_of_rows h (row := 124) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_4
  simp only [← c5148, ← c5149, ← c5150] at e_124_4
  have hr := e_124_4
  simp only [e_124_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f623 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 125 12) = spongeHash perm [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262)] 0 ∧ a (.wire 125 13) = spongeHash perm [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262)] 1 ∧ a (.wire 125 14) = spongeHash perm [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262)] 2 ∧ a (.wire 125 15) = spongeHash perm [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262)] 3) := by
  have c5151 := (privateBatchWrapper8_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5152 := (privateBatchWrapper8_copies161 a h).1
  have c5153 := (privateBatchWrapper8_copies161 a h).2.1
  have c5154 := (privateBatchWrapper8_copies161 a h).2.2.1
  have c5155 := (privateBatchWrapper8_copies161 a h).2.2.2.1
  have c5156 := (privateBatchWrapper8_copies161 a h).2.2.2.2.1
  have c5157 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.1
  have c5158 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.1
  have c5159 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.1
  have c5160 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.1
  have c5161 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.1
  have c5162 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 125 = addBlock (fun _ => 0) [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), 1, 0, 0, 0] :=
    poseidon2In_first c5151.symm c5152.symm c5153.symm c5154.symm (c5155.symm.trans k3) (c5156.symm.trans k2) (c5157.symm.trans k2) (c5158.symm.trans k2) (c5159.symm.trans k2) (c5160.symm.trans k2) (c5161.symm.trans k2) (c5162.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 125) rfl rfl hin0
  have hpad : pad10 [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262)] =
      [a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f624 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 126 12) = spongeHash perm [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15)] 0 ∧ a (.wire 126 13) = spongeHash perm [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15)] 1 ∧ a (.wire 126 14) = spongeHash perm [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15)] 2 ∧ a (.wire 126 15) = spongeHash perm [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15)] 3) := by
  have c5163 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5164 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5165 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5166 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5167 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5168 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5169 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5170 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5171 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5172 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5173 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5174 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 126 = addBlock (fun _ => 0) [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15), 1, 0, 0, 0] :=
    poseidon2In_first c5163.symm c5164.symm c5165.symm c5166.symm (c5167.symm.trans k3) (c5168.symm.trans k2) (c5169.symm.trans k2) (c5170.symm.trans k2) (c5171.symm.trans k2) (c5172.symm.trans k2) (c5173.symm.trans k2) (c5174.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 126) rfl rfl hin0
  have hpad : pad10 [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15)] =
      [a (.wire 125 12), a (.wire 125 13), a (.wire 125 14), a (.wire 125 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f625 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 27) = bselect (a (.wire 4 59)) (a (.wire 126 12)) (a (.virt 25)) := by
  have c5175 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5176 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5177 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5178 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5179 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5180 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_5 := arithEq_of_rows h (row := 124) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_5
  simp only [← c5175, ← c5176, ← c5177] at e_124_5
  have e_124_6 := arithEq_of_rows h (row := 124) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_6
  simp only [← c5178, ← c5179, ← c5180] at e_124_6
  have hr := e_124_6
  simp only [e_124_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f626 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 35) = bselect (a (.wire 4 59)) (a (.wire 126 13)) (a (.virt 26)) := by
  have c5181 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5182 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5183 := (privateBatchWrapper8_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5184 := (privateBatchWrapper8_copies162 a h).1
  have c5185 := (privateBatchWrapper8_copies162 a h).2.1
  have c5186 := (privateBatchWrapper8_copies162 a h).2.2.1
  have e_124_7 := arithEq_of_rows h (row := 124) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_7
  simp only [← c5181, ← c5182, ← c5183] at e_124_7
  have e_124_8 := arithEq_of_rows h (row := 124) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_8
  simp only [← c5184, ← c5185, ← c5186] at e_124_8
  have hr := e_124_8
  simp only [e_124_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f627 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 43) = bselect (a (.wire 4 59)) (a (.wire 126 14)) (a (.virt 27)) := by
  have c5187 := (privateBatchWrapper8_copies162 a h).2.2.2.1
  have c5188 := (privateBatchWrapper8_copies162 a h).2.2.2.2.1
  have c5189 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.1
  have c5190 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.1
  have c5191 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.1
  have c5192 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.1
  have e_124_9 := arithEq_of_rows h (row := 124) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_9
  simp only [← c5187, ← c5188, ← c5189] at e_124_9
  have e_124_10 := arithEq_of_rows h (row := 124) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_10
  simp only [← c5190, ← c5191, ← c5192] at e_124_10
  have hr := e_124_10
  simp only [e_124_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f628 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 51) = bselect (a (.wire 4 59)) (a (.wire 126 15)) (a (.virt 28)) := by
  have c5193 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.1
  have c5194 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5195 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5196 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5197 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5198 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_11 := arithEq_of_rows h (row := 124) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_11
  simp only [← c5193, ← c5194, ← c5195] at e_124_11
  have e_124_12 := arithEq_of_rows h (row := 124) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_12
  simp only [← c5196, ← c5197, ← c5198] at e_124_12
  have hr := e_124_12
  simp only [e_124_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f629 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 127 12) = spongeHash perm [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266)] 0 ∧ a (.wire 127 13) = spongeHash perm [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266)] 1 ∧ a (.wire 127 14) = spongeHash perm [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266)] 2 ∧ a (.wire 127 15) = spongeHash perm [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266)] 3) := by
  have c5199 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5200 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5201 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5202 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5203 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5204 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5205 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5206 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5207 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5208 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5209 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5210 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 127 = addBlock (fun _ => 0) [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), 1, 0, 0, 0] :=
    poseidon2In_first c5199.symm c5200.symm c5201.symm c5202.symm (c5203.symm.trans k3) (c5204.symm.trans k2) (c5205.symm.trans k2) (c5206.symm.trans k2) (c5207.symm.trans k2) (c5208.symm.trans k2) (c5209.symm.trans k2) (c5210.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 127) rfl rfl hin0
  have hpad : pad10 [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266)] =
      [a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f630 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 128 12) = spongeHash perm [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15)] 0 ∧ a (.wire 128 13) = spongeHash perm [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15)] 1 ∧ a (.wire 128 14) = spongeHash perm [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15)] 2 ∧ a (.wire 128 15) = spongeHash perm [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15)] 3) := by
  have c5211 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5212 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5213 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5214 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5215 := (privateBatchWrapper8_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5216 := (privateBatchWrapper8_copies163 a h).1
  have c5217 := (privateBatchWrapper8_copies163 a h).2.1
  have c5218 := (privateBatchWrapper8_copies163 a h).2.2.1
  have c5219 := (privateBatchWrapper8_copies163 a h).2.2.2.1
  have c5220 := (privateBatchWrapper8_copies163 a h).2.2.2.2.1
  have c5221 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.1
  have c5222 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 128 = addBlock (fun _ => 0) [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15), 1, 0, 0, 0] :=
    poseidon2In_first c5211.symm c5212.symm c5213.symm c5214.symm (c5215.symm.trans k3) (c5216.symm.trans k2) (c5217.symm.trans k2) (c5218.symm.trans k2) (c5219.symm.trans k2) (c5220.symm.trans k2) (c5221.symm.trans k2) (c5222.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 128) rfl rfl hin0
  have hpad : pad10 [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15)] =
      [a (.wire 127 12), a (.wire 127 13), a (.wire 127 14), a (.wire 127 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f631 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 124 59) = bselect (a (.wire 6 43)) (a (.wire 128 12)) (a (.virt 36)) := by
  have c5223 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.1
  have c5224 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.1
  have c5225 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.1
  have c5226 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5227 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5228 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_13 := arithEq_of_rows h (row := 124) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_13
  simp only [← c5223, ← c5224, ← c5225] at e_124_13
  have e_124_14 := arithEq_of_rows h (row := 124) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_14
  simp only [← c5226, ← c5227, ← c5228] at e_124_14
  have hr := e_124_14
  simp only [e_124_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f632 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 7) = bselect (a (.wire 6 43)) (a (.wire 128 13)) (a (.virt 37)) := by
  have c5229 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5230 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5231 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5232 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5233 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5234 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_0 := arithEq_of_rows h (row := 129) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_0
  simp only [← c5229, ← c5230, ← c5231] at e_129_0
  have e_129_1 := arithEq_of_rows h (row := 129) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_1
  simp only [← c5232, ← c5233, ← c5234] at e_129_1
  have hr := e_129_1
  simp only [e_129_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f633 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 15) = bselect (a (.wire 6 43)) (a (.wire 128 14)) (a (.virt 38)) := by
  have c5235 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5236 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5237 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5238 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5239 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5240 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_2 := arithEq_of_rows h (row := 129) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_2
  simp only [← c5235, ← c5236, ← c5237] at e_129_2
  have e_129_3 := arithEq_of_rows h (row := 129) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_3
  simp only [← c5238, ← c5239, ← c5240] at e_129_3
  have hr := e_129_3
  simp only [e_129_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f634 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 23) = bselect (a (.wire 6 43)) (a (.wire 128 15)) (a (.virt 39)) := by
  have c5241 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5242 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5243 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5244 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5245 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5246 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_4 := arithEq_of_rows h (row := 129) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_4
  simp only [← c5241, ← c5242, ← c5243] at e_129_4
  have e_129_5 := arithEq_of_rows h (row := 129) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_5
  simp only [← c5244, ← c5245, ← c5246] at e_129_5
  have hr := e_129_5
  simp only [e_129_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f635 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 130 12) = spongeHash perm [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270)] 0 ∧ a (.wire 130 13) = spongeHash perm [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270)] 1 ∧ a (.wire 130 14) = spongeHash perm [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270)] 2 ∧ a (.wire 130 15) = spongeHash perm [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270)] 3) := by
  have c5247 := (privateBatchWrapper8_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5248 := (privateBatchWrapper8_copies164 a h).1
  have c5249 := (privateBatchWrapper8_copies164 a h).2.1
  have c5250 := (privateBatchWrapper8_copies164 a h).2.2.1
  have c5251 := (privateBatchWrapper8_copies164 a h).2.2.2.1
  have c5252 := (privateBatchWrapper8_copies164 a h).2.2.2.2.1
  have c5253 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.1
  have c5254 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.1
  have c5255 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.1
  have c5256 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.1
  have c5257 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.1
  have c5258 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 130 = addBlock (fun _ => 0) [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), 1, 0, 0, 0] :=
    poseidon2In_first c5247.symm c5248.symm c5249.symm c5250.symm (c5251.symm.trans k3) (c5252.symm.trans k2) (c5253.symm.trans k2) (c5254.symm.trans k2) (c5255.symm.trans k2) (c5256.symm.trans k2) (c5257.symm.trans k2) (c5258.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 130) rfl rfl hin0
  have hpad : pad10 [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270)] =
      [a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f636 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 131 12) = spongeHash perm [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15)] 0 ∧ a (.wire 131 13) = spongeHash perm [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15)] 1 ∧ a (.wire 131 14) = spongeHash perm [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15)] 2 ∧ a (.wire 131 15) = spongeHash perm [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15)] 3) := by
  have c5259 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5260 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5261 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5262 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5263 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5264 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5265 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5266 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5267 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5268 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5269 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5270 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 131 = addBlock (fun _ => 0) [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15), 1, 0, 0, 0] :=
    poseidon2In_first c5259.symm c5260.symm c5261.symm c5262.symm (c5263.symm.trans k3) (c5264.symm.trans k2) (c5265.symm.trans k2) (c5266.symm.trans k2) (c5267.symm.trans k2) (c5268.symm.trans k2) (c5269.symm.trans k2) (c5270.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 131) rfl rfl hin0
  have hpad : pad10 [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15)] =
      [a (.wire 130 12), a (.wire 130 13), a (.wire 130 14), a (.wire 130 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f637 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 31) = bselect (a (.wire 8 27)) (a (.wire 131 12)) (a (.virt 47)) := by
  have c5271 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5272 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5273 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5274 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5275 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5276 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_6 := arithEq_of_rows h (row := 129) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_6
  simp only [← c5271, ← c5272, ← c5273] at e_129_6
  have e_129_7 := arithEq_of_rows h (row := 129) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_7
  simp only [← c5274, ← c5275, ← c5276] at e_129_7
  have hr := e_129_7
  simp only [e_129_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f638 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 39) = bselect (a (.wire 8 27)) (a (.wire 131 13)) (a (.virt 48)) := by
  have c5277 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5278 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5279 := (privateBatchWrapper8_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5280 := (privateBatchWrapper8_copies165 a h).1
  have c5281 := (privateBatchWrapper8_copies165 a h).2.1
  have c5282 := (privateBatchWrapper8_copies165 a h).2.2.1
  have e_129_8 := arithEq_of_rows h (row := 129) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_8
  simp only [← c5277, ← c5278, ← c5279] at e_129_8
  have e_129_9 := arithEq_of_rows h (row := 129) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_9
  simp only [← c5280, ← c5281, ← c5282] at e_129_9
  have hr := e_129_9
  simp only [e_129_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f639 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 47) = bselect (a (.wire 8 27)) (a (.wire 131 14)) (a (.virt 49)) := by
  have c5283 := (privateBatchWrapper8_copies165 a h).2.2.2.1
  have c5284 := (privateBatchWrapper8_copies165 a h).2.2.2.2.1
  have c5285 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.1
  have c5286 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.1
  have c5287 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.1
  have c5288 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.1
  have e_129_10 := arithEq_of_rows h (row := 129) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_10
  simp only [← c5283, ← c5284, ← c5285] at e_129_10
  have e_129_11 := arithEq_of_rows h (row := 129) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_11
  simp only [← c5286, ← c5287, ← c5288] at e_129_11
  have hr := e_129_11
  simp only [e_129_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f640 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 129 55) = bselect (a (.wire 8 27)) (a (.wire 131 15)) (a (.virt 50)) := by
  have c5289 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.1
  have c5290 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5291 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5292 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5293 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5294 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_12 := arithEq_of_rows h (row := 129) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_12
  simp only [← c5289, ← c5290, ← c5291] at e_129_12
  have e_129_13 := arithEq_of_rows h (row := 129) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_13
  simp only [← c5292, ← c5293, ← c5294] at e_129_13
  have hr := e_129_13
  simp only [e_129_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f641 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 132 12) = spongeHash perm [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274)] 0 ∧ a (.wire 132 13) = spongeHash perm [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274)] 1 ∧ a (.wire 132 14) = spongeHash perm [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274)] 2 ∧ a (.wire 132 15) = spongeHash perm [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274)] 3) := by
  have c5295 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5296 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5297 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5298 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5299 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5300 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5301 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5302 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5303 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5304 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5305 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5306 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 132 = addBlock (fun _ => 0) [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), 1, 0, 0, 0] :=
    poseidon2In_first c5295.symm c5296.symm c5297.symm c5298.symm (c5299.symm.trans k3) (c5300.symm.trans k2) (c5301.symm.trans k2) (c5302.symm.trans k2) (c5303.symm.trans k2) (c5304.symm.trans k2) (c5305.symm.trans k2) (c5306.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 132) rfl rfl hin0
  have hpad : pad10 [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274)] =
      [a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f642 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 133 12) = spongeHash perm [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15)] 0 ∧ a (.wire 133 13) = spongeHash perm [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15)] 1 ∧ a (.wire 133 14) = spongeHash perm [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15)] 2 ∧ a (.wire 133 15) = spongeHash perm [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15)] 3) := by
  have c5307 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5308 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5309 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5310 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5311 := (privateBatchWrapper8_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5312 := (privateBatchWrapper8_copies166 a h).1
  have c5313 := (privateBatchWrapper8_copies166 a h).2.1
  have c5314 := (privateBatchWrapper8_copies166 a h).2.2.1
  have c5315 := (privateBatchWrapper8_copies166 a h).2.2.2.1
  have c5316 := (privateBatchWrapper8_copies166 a h).2.2.2.2.1
  have c5317 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.1
  have c5318 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 133 = addBlock (fun _ => 0) [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15), 1, 0, 0, 0] :=
    poseidon2In_first c5307.symm c5308.symm c5309.symm c5310.symm (c5311.symm.trans k3) (c5312.symm.trans k2) (c5313.symm.trans k2) (c5314.symm.trans k2) (c5315.symm.trans k2) (c5316.symm.trans k2) (c5317.symm.trans k2) (c5318.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 133) rfl rfl hin0
  have hpad : pad10 [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15)] =
      [a (.wire 132 12), a (.wire 132 13), a (.wire 132 14), a (.wire 132 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f643 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 134 3) = bselect (a (.wire 10 11)) (a (.wire 133 12)) (a (.virt 58)) := by
  have c5319 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.1
  have c5320 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.1
  have c5321 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.1
  have c5322 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5323 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5324 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_14 := arithEq_of_rows h (row := 129) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_14
  simp only [← c5319, ← c5320, ← c5321] at e_129_14
  have e_134_0 := arithEq_of_rows h (row := 134) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_0
  simp only [← c5322, ← c5323, ← c5324] at e_134_0
  have hr := e_134_0
  simp only [e_129_14] at hr
  simp only [bselect]
  linear_combination hr

end Plonky2Spec.Generated
