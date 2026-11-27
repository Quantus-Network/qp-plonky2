/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PrivateBatchWrapper4`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge
import Plonky2Spec.Generated.PrivateBatchWrapper4.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem privateBatchWrapper4_f0 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5)) (a (.virt 11)) (a (.virt 84)) (a (.virt 85)) := by
  have c0 := (privateBatchWrapper4_copies0 a h).1
  have c1 := (privateBatchWrapper4_copies0 a h).2.1
  have c2 := (privateBatchWrapper4_copies0 a h).2.2.1
  have c3 := (privateBatchWrapper4_copies0 a h).2.2.2.1
  have c4 := (privateBatchWrapper4_copies0 a h).2.2.2.2.1
  have c5 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.1
  have c6 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.1
  have c7 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_6
  simp only [← c3, ← c4, ← c5] at e_1_6
  have e_1_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_7
  simp only [← c6, ← c7, ← c8] at e_1_7
  have e_2_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c0, k3, ← c1, k3, ← c2] at e_2_0
  have e_2_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c9, ← c10, k3, ← c11] at e_2_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_6
    linear_combination c12.trans k1 - hc
  · have hc := e_2_1
    simp only [e_2_0, e_1_7] at hc
    linear_combination c13.trans k1 - hc

theorem privateBatchWrapper4_f1 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 6)) (a (.virt 11)) (a (.virt 86)) (a (.virt 87)) := by
  have c14 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c17, ← c18, ← c19] at e_1_8
  have e_1_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c20, ← c21, ← c22] at e_1_9
  have e_2_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c14, k3, ← c15, k3, ← c16] at e_2_2
  have e_2_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c23, ← c24, k3, ← c25] at e_2_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_8
    linear_combination c26.trans k1 - hc
  · have hc := e_2_3
    simp only [e_2_2, e_1_9] at hc
    linear_combination c27.trans k1 - hc

theorem privateBatchWrapper4_f2 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 7)) (a (.virt 11)) (a (.virt 88)) (a (.virt 89)) := by
  have c28 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (privateBatchWrapper4_copies1 a h).1
  have c33 := (privateBatchWrapper4_copies1 a h).2.1
  have c34 := (privateBatchWrapper4_copies1 a h).2.2.1
  have c35 := (privateBatchWrapper4_copies1 a h).2.2.2.1
  have c36 := (privateBatchWrapper4_copies1 a h).2.2.2.2.1
  have c37 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.1
  have c38 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.1
  have c39 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c31, ← c32, ← c33] at e_1_10
  have e_1_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_11
  simp only [← c34, ← c35, ← c36] at e_1_11
  have e_2_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c28, k3, ← c29, k3, ← c30] at e_2_4
  have e_2_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c37, ← c38, k3, ← c39] at e_2_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_10
    linear_combination c40.trans k1 - hc
  · have hc := e_2_5
    simp only [e_2_4, e_1_11] at hc
    linear_combination c41.trans k1 - hc

theorem privateBatchWrapper4_f3 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 8)) (a (.virt 11)) (a (.virt 90)) (a (.virt 91)) := by
  have c42 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c45, ← c46, ← c47] at e_1_12
  have e_1_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c48, ← c49, ← c50] at e_1_13
  have e_2_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c42, k3, ← c43, k3, ← c44] at e_2_6
  have e_2_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c51, ← c52, k3, ← c53] at e_2_7
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_12
    linear_combination c54.trans k1 - hc
  · have hc := e_2_7
    simp only [e_2_6, e_1_13] at hc
    linear_combination c55.trans k1 - hc

theorem privateBatchWrapper4_f4 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 1 59) = band (a (.virt 84)) (a (.virt 86)) := by
  have c56 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c56, ← c57, ← c58] at e_1_14
  have hr := e_1_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f5 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 3 3) = band (a (.virt 88)) (a (.virt 90)) := by
  have c59 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c59, ← c60, ← c61] at e_3_0
  have hr := e_3_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f6 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 3 7) = band (a (.wire 1 59)) (a (.wire 3 3)) := by
  have c62 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (privateBatchWrapper4_copies2 a h).1
  have e_3_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c62, ← c63, ← c64] at e_3_1
  have hr := e_3_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f7 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 19)) (a (.virt 11)) (a (.virt 92)) (a (.virt 93)) := by
  have c65 := (privateBatchWrapper4_copies2 a h).2.1
  have c66 := (privateBatchWrapper4_copies2 a h).2.2.1
  have c67 := (privateBatchWrapper4_copies2 a h).2.2.2.1
  have c68 := (privateBatchWrapper4_copies2 a h).2.2.2.2.1
  have c69 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.1
  have c70 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.1
  have c71 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c65, k3, ← c66, k3, ← c67] at e_2_8
  have e_2_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c74, ← c75, k3, ← c76] at e_2_9
  have e_3_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c68, ← c69, ← c70] at e_3_2
  have e_3_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c71, ← c72, ← c73] at e_3_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_2
    linear_combination c77.trans k1 - hc
  · have hc := e_2_9
    simp only [e_2_8, e_3_3] at hc
    linear_combination c78.trans k1 - hc

theorem privateBatchWrapper4_f8 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 20)) (a (.virt 11)) (a (.virt 94)) (a (.virt 95)) := by
  have c79 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c79, k3, ← c80, k3, ← c81] at e_2_10
  have e_2_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c88, ← c89, k3, ← c90] at e_2_11
  have e_3_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c82, ← c83, ← c84] at e_3_4
  have e_3_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c85, ← c86, ← c87] at e_3_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_4
    linear_combination c91.trans k1 - hc
  · have hc := e_2_11
    simp only [e_2_10, e_3_5] at hc
    linear_combination c92.trans k1 - hc

theorem privateBatchWrapper4_f9 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 21)) (a (.virt 11)) (a (.virt 96)) (a (.virt 97)) := by
  have c93 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (privateBatchWrapper4_copies3 a h).1
  have c97 := (privateBatchWrapper4_copies3 a h).2.1
  have c98 := (privateBatchWrapper4_copies3 a h).2.2.1
  have c99 := (privateBatchWrapper4_copies3 a h).2.2.2.1
  have c100 := (privateBatchWrapper4_copies3 a h).2.2.2.2.1
  have c101 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.1
  have c102 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.1
  have c103 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c93, k3, ← c94, k3, ← c95] at e_2_12
  have e_2_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c102, ← c103, k3, ← c104] at e_2_13
  have e_3_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c96, ← c97, ← c98] at e_3_6
  have e_3_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c99, ← c100, ← c101] at e_3_7
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_6
    linear_combination c105.trans k1 - hc
  · have hc := e_2_13
    simp only [e_2_12, e_3_7] at hc
    linear_combination c106.trans k1 - hc

theorem privateBatchWrapper4_f10 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 22)) (a (.virt 11)) (a (.virt 98)) (a (.virt 99)) := by
  have c107 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c107, k3, ← c108, k3, ← c109] at e_2_14
  have e_3_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c110, ← c111, ← c112] at e_3_8
  have e_3_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c113, ← c114, ← c115] at e_3_9
  have e_4_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c116, ← c117, k3, ← c118] at e_4_0
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_8
    linear_combination c119.trans k1 - hc
  · have hc := e_4_0
    simp only [e_2_14, e_3_9] at hc
    linear_combination c120.trans k1 - hc

theorem privateBatchWrapper4_f11 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 3 43) = band (a (.virt 92)) (a (.virt 94)) := by
  have c121 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c121, ← c122, ← c123] at e_3_10
  have hr := e_3_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f12 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 3 47) = band (a (.virt 96)) (a (.virt 98)) := by
  have c124 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c124, ← c125, ← c126] at e_3_11
  have hr := e_3_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f13 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 3 51) = band (a (.wire 3 43)) (a (.wire 3 47)) := by
  have c127 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (privateBatchWrapper4_copies4 a h).1
  have c129 := (privateBatchWrapper4_copies4 a h).2.1
  have e_3_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c127, ← c128, ← c129] at e_3_12
  have hr := e_3_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f14 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 30)) (a (.virt 11)) (a (.virt 100)) (a (.virt 101)) := by
  have c130 := (privateBatchWrapper4_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper4_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper4_copies4 a h).2.2.2.2.1
  have c133 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.1
  have c134 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.1
  have c135 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_3_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c133, ← c134, ← c135] at e_3_13
  have e_3_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c136, ← c137, ← c138] at e_3_14
  have e_4_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c130, k3, ← c131, k3, ← c132] at e_4_1
  have e_4_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c139, ← c140, k3, ← c141] at e_4_2
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_13
    linear_combination c142.trans k1 - hc
  · have hc := e_4_2
    simp only [e_4_1, e_3_14] at hc
    linear_combination c143.trans k1 - hc

theorem privateBatchWrapper4_f15 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 31)) (a (.virt 11)) (a (.virt 102)) (a (.virt 103)) := by
  have c144 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c145 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c148 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c151 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c144, k3, ← c145, k3, ← c146] at e_4_3
  have e_4_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c153, ← c154, k3, ← c155] at e_4_4
  have e_5_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c147, ← c148, ← c149] at e_5_0
  have e_5_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c150, ← c151, ← c152] at e_5_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_5_0
    linear_combination c156.trans k1 - hc
  · have hc := e_4_4
    simp only [e_4_3, e_5_1] at hc
    linear_combination c157.trans k1 - hc

theorem privateBatchWrapper4_f16 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 32)) (a (.virt 11)) (a (.virt 104)) (a (.virt 105)) := by
  have c158 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (privateBatchWrapper4_copies5 a h).1
  have c161 := (privateBatchWrapper4_copies5 a h).2.1
  have c162 := (privateBatchWrapper4_copies5 a h).2.2.1
  have c163 := (privateBatchWrapper4_copies5 a h).2.2.2.1
  have c164 := (privateBatchWrapper4_copies5 a h).2.2.2.2.1
  have c165 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.1
  have c166 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.1
  have c167 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c158, k3, ← c159, k3, ← c160] at e_4_5
  have e_4_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c167, ← c168, k3, ← c169] at e_4_6
  have e_5_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c161, ← c162, ← c163] at e_5_2
  have e_5_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c164, ← c165, ← c166] at e_5_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_5_2
    linear_combination c170.trans k1 - hc
  · have hc := e_4_6
    simp only [e_4_5, e_5_3] at hc
    linear_combination c171.trans k1 - hc

theorem privateBatchWrapper4_f17 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 33)) (a (.virt 11)) (a (.virt 106)) (a (.virt 107)) := by
  have c172 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c172, k3, ← c173, k3, ← c174] at e_4_7
  have e_4_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c181, ← c182, k3, ← c183] at e_4_8
  have e_5_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c175, ← c176, ← c177] at e_5_4
  have e_5_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c178, ← c179, ← c180] at e_5_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_5_4
    linear_combination c184.trans k1 - hc
  · have hc := e_4_8
    simp only [e_4_7, e_5_5] at hc
    linear_combination c185.trans k1 - hc

theorem privateBatchWrapper4_f18 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 27) = band (a (.virt 100)) (a (.virt 102)) := by
  have c186 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c186, ← c187, ← c188] at e_5_6
  have hr := e_5_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f19 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 31) = band (a (.virt 104)) (a (.virt 106)) := by
  have c189 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_5_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c189, ← c190, ← c191] at e_5_7
  have hr := e_5_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f20 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 35) = band (a (.wire 5 27)) (a (.wire 5 31)) := by
  have c192 := (privateBatchWrapper4_copies6 a h).1
  have c193 := (privateBatchWrapper4_copies6 a h).2.1
  have c194 := (privateBatchWrapper4_copies6 a h).2.2.1
  have e_5_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c192, ← c193, ← c194] at e_5_8
  have hr := e_5_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f21 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 41)) (a (.virt 11)) (a (.virt 108)) (a (.virt 109)) := by
  have c195 := (privateBatchWrapper4_copies6 a h).2.2.2.1
  have c196 := (privateBatchWrapper4_copies6 a h).2.2.2.2.1
  have c197 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.1
  have c198 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.1
  have c199 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c195, k3, ← c196, k3, ← c197] at e_4_9
  have e_4_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c204, ← c205, k3, ← c206] at e_4_10
  have e_5_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c198, ← c199, ← c200] at e_5_9
  have e_5_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c201, ← c202, ← c203] at e_5_10
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_5_9
    linear_combination c207.trans k1 - hc
  · have hc := e_4_10
    simp only [e_4_9, e_5_10] at hc
    linear_combination c208.trans k1 - hc

theorem privateBatchWrapper4_f22 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 42)) (a (.virt 11)) (a (.virt 110)) (a (.virt 111)) := by
  have c209 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c209, k3, ← c210, k3, ← c211] at e_4_11
  have e_4_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c218, ← c219, k3, ← c220] at e_4_12
  have e_5_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c212, ← c213, ← c214] at e_5_11
  have e_5_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c215, ← c216, ← c217] at e_5_12
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_5_11
    linear_combination c221.trans k1 - hc
  · have hc := e_4_12
    simp only [e_4_11, e_5_12] at hc
    linear_combination c222.trans k1 - hc

theorem privateBatchWrapper4_f23 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 43)) (a (.virt 11)) (a (.virt 112)) (a (.virt 113)) := by
  have c223 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (privateBatchWrapper4_copies7 a h).1
  have c225 := (privateBatchWrapper4_copies7 a h).2.1
  have c226 := (privateBatchWrapper4_copies7 a h).2.2.1
  have c227 := (privateBatchWrapper4_copies7 a h).2.2.2.1
  have c228 := (privateBatchWrapper4_copies7 a h).2.2.2.2.1
  have c229 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.1
  have c230 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.1
  have c231 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c223, k3, ← c224, k3, ← c225] at e_4_13
  have e_4_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c232, ← c233, k3, ← c234] at e_4_14
  have e_5_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c226, ← c227, ← c228] at e_5_13
  have e_5_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c229, ← c230, ← c231] at e_5_14
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_5_13
    linear_combination c235.trans k1 - hc
  · have hc := e_4_14
    simp only [e_4_13, e_5_14] at hc
    linear_combination c236.trans k1 - hc

theorem privateBatchWrapper4_f24 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 44)) (a (.virt 11)) (a (.virt 114)) (a (.virt 115)) := by
  have c237 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c237, k3, ← c238, k3, ← c239] at e_6_0
  have e_6_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c246, ← c247, k3, ← c248] at e_6_1
  have e_7_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c240, ← c241, ← c242] at e_7_0
  have e_7_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c243, ← c244, ← c245] at e_7_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_7_0
    linear_combination c249.trans k1 - hc
  · have hc := e_6_1
    simp only [e_6_0, e_7_1] at hc
    linear_combination c250.trans k1 - hc

theorem privateBatchWrapper4_f25 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 11) = band (a (.virt 108)) (a (.virt 110)) := by
  have c251 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c251, ← c252, ← c253] at e_7_2
  have hr := e_7_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f26 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 15) = band (a (.virt 112)) (a (.virt 114)) := by
  have c254 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (privateBatchWrapper4_copies8 a h).1
  have e_7_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c254, ← c255, ← c256] at e_7_3
  have hr := e_7_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f27 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 19) = band (a (.wire 7 11)) (a (.wire 7 15)) := by
  have c257 := (privateBatchWrapper4_copies8 a h).2.1
  have c258 := (privateBatchWrapper4_copies8 a h).2.2.1
  have c259 := (privateBatchWrapper4_copies8 a h).2.2.2.1
  have e_7_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c257, ← c258, ← c259] at e_7_4
  have hr := e_7_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f28 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 11) = bnot (a (.wire 3 7)) := by
  have c260 := (privateBatchWrapper4_copies8 a h).2.2.2.2.1
  have c261 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c260, k3, ← c261, k3, ← c262] at e_6_2
  have hr := e_6_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f29 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 13) = bnot (a (.virt 11)) := by
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [bnot, k1, k3]
  ring

theorem privateBatchWrapper4_f30 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 11) = band (a (.wire 6 11)) (a (.virt 13)) := by
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [band, k3]
  ring

theorem privateBatchWrapper4_f31 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 15) = bselect (a (.wire 6 11)) (a (.virt 5)) (a (.virt 11)) := by
  have c263 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c263, ← c264, ← c265, k1] at e_6_3
  have hr := e_6_3
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f32 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 19) = bselect (a (.wire 6 11)) (a (.virt 6)) (a (.virt 11)) := by
  have c266 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c266, ← c267, ← c268, k1] at e_6_4
  have hr := e_6_4
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f33 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 23) = bselect (a (.wire 6 11)) (a (.virt 7)) (a (.virt 11)) := by
  have c269 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c269, ← c270, ← c271, k1] at e_6_5
  have hr := e_6_5
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f34 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 27) = bselect (a (.wire 6 11)) (a (.virt 8)) (a (.virt 11)) := by
  have c272 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c272, ← c273, ← c274, k1] at e_6_6
  have hr := e_6_6
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f35 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 31) = bselect (a (.wire 6 11)) (a (.virt 9)) (a (.virt 11)) := by
  have c275 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c275, ← c276, ← c277, k1] at e_6_7
  have hr := e_6_7
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f36 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 11) = bor (a (.virt 11)) (a (.wire 6 11)) := by
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [bor, k1]
  ring

theorem privateBatchWrapper4_f37 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 35) = bnot (a (.wire 3 51)) := by
  have c278 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c278, k3, ← c279, k3, ← c280] at e_6_8
  have hr := e_6_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f38 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 39) = bnot (a (.wire 6 11)) := by
  have c281 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c281, k3, ← c282, k3, ← c283] at e_6_9
  have hr := e_6_9
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f39 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 23) = band (a (.wire 6 35)) (a (.wire 6 39)) := by
  have c284 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c284, ← c285, ← c286] at e_7_5
  have hr := e_7_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f40 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 47) = bselect (a (.wire 7 23)) (a (.virt 19)) (a (.wire 6 15)) := by
  have c287 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (privateBatchWrapper4_copies9 a h).1
  have c289 := (privateBatchWrapper4_copies9 a h).2.1
  have c290 := (privateBatchWrapper4_copies9 a h).2.2.1
  have c291 := (privateBatchWrapper4_copies9 a h).2.2.2.1
  have c292 := (privateBatchWrapper4_copies9 a h).2.2.2.2.1
  have e_6_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c287, ← c288, ← c289] at e_6_10
  have e_6_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c290, ← c291, ← c292] at e_6_11
  have hr := e_6_11
  simp only [e_6_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f41 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 55) = bselect (a (.wire 7 23)) (a (.virt 20)) (a (.wire 6 19)) := by
  have c293 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.1
  have c294 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.1
  have c295 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_6_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c293, ← c294, ← c295] at e_6_12
  have e_6_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c296, ← c297, ← c298] at e_6_13
  have hr := e_6_13
  simp only [e_6_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f42 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 3) = bselect (a (.wire 7 23)) (a (.virt 21)) (a (.wire 6 23)) := by
  have c299 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c299, ← c300, ← c301] at e_6_14
  have e_8_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c302, ← c303, ← c304] at e_8_0
  have hr := e_8_0
  simp only [e_6_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f43 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 11) = bselect (a (.wire 7 23)) (a (.virt 22)) (a (.wire 6 27)) := by
  have c305 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c305, ← c306, ← c307] at e_8_1
  have e_8_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c308, ← c309, ← c310] at e_8_2
  have hr := e_8_2
  simp only [e_8_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f44 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 19) = bselect (a (.wire 7 23)) (a (.virt 23)) (a (.wire 6 31)) := by
  have c311 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c311, ← c312, ← c313] at e_8_3
  have e_8_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c314, ← c315, ← c316] at e_8_4
  have hr := e_8_4
  simp only [e_8_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f45 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 31) = bor (a (.wire 6 11)) (a (.wire 6 35)) := by
  have c317 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (privateBatchWrapper4_copies10 a h).1
  have c321 := (privateBatchWrapper4_copies10 a h).2.1
  have c322 := (privateBatchWrapper4_copies10 a h).2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_7
  simp only [← c320, ← c321, k3, ← c322] at e_0_7
  have e_9_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c317, ← c318, ← c319] at e_9_0
  have hr := e_0_7
  simp only [e_9_0] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f46 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 23) = bnot (a (.wire 5 35)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c323, k3, ← c324, k3, ← c325] at e_8_5
  have hr := e_8_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f47 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 27) = bnot (a (.wire 0 31)) := by
  have c326 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.1
  have c327 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c326, k3, ← c327, k3, ← c328] at e_8_6
  have hr := e_8_6
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f48 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 27) = band (a (.wire 8 23)) (a (.wire 8 27)) := by
  have c329 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c329, ← c330, ← c331] at e_7_6
  have hr := e_7_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f49 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 35) = bselect (a (.wire 7 27)) (a (.virt 30)) (a (.wire 6 47)) := by
  have c332 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c332, ← c333, ← c334] at e_8_7
  have e_8_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c335, ← c336, ← c337] at e_8_8
  have hr := e_8_8
  simp only [e_8_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f50 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 43) = bselect (a (.wire 7 27)) (a (.virt 31)) (a (.wire 6 55)) := by
  have c338 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c338, ← c339, ← c340] at e_8_9
  have e_8_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c341, ← c342, ← c343] at e_8_10
  have hr := e_8_10
  simp only [e_8_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f51 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 51) = bselect (a (.wire 7 27)) (a (.virt 32)) (a (.wire 8 3)) := by
  have c344 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c344, ← c345, ← c346] at e_8_11
  have e_8_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c347, ← c348, ← c349] at e_8_12
  have hr := e_8_12
  simp only [e_8_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f52 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 59) = bselect (a (.wire 7 27)) (a (.virt 33)) (a (.wire 8 11)) := by
  have c350 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (privateBatchWrapper4_copies11 a h).1
  have c353 := (privateBatchWrapper4_copies11 a h).2.1
  have c354 := (privateBatchWrapper4_copies11 a h).2.2.1
  have c355 := (privateBatchWrapper4_copies11 a h).2.2.2.1
  have e_8_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c350, ← c351, ← c352] at e_8_13
  have e_8_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c353, ← c354, ← c355] at e_8_14
  have hr := e_8_14
  simp only [e_8_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f53 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 7) = bselect (a (.wire 7 27)) (a (.virt 34)) (a (.wire 8 19)) := by
  have c356 := (privateBatchWrapper4_copies11 a h).2.2.2.2.1
  have c357 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.1
  have c358 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.1
  have c359 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have e_10_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c356, ← c357, ← c358] at e_10_0
  have e_10_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c359, ← c360, ← c361] at e_10_1
  have hr := e_10_1
  simp only [e_10_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f54 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 35) = bor (a (.wire 0 31)) (a (.wire 8 23)) := by
  have c362 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_8
  simp only [← c365, ← c366, k3, ← c367] at e_0_8
  have e_9_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c362, ← c363, ← c364] at e_9_1
  have hr := e_0_8
  simp only [e_9_1] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f55 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 7 19)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k3, ← c369, k3, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f56 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 15) = bnot (a (.wire 0 35)) := by
  have c371 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c371, k3, ← c372, k3, ← c373] at e_10_3
  have hr := e_10_3
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f57 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 31) = band (a (.wire 10 11)) (a (.wire 10 15)) := by
  have c374 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c374, ← c375, ← c376] at e_7_7
  have hr := e_7_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f58 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 23) = bselect (a (.wire 7 31)) (a (.virt 41)) (a (.wire 8 35)) := by
  have c377 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c377, ← c378, ← c379] at e_10_4
  have e_10_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c380, ← c381, ← c382] at e_10_5
  have hr := e_10_5
  simp only [e_10_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f59 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 31) = bselect (a (.wire 7 31)) (a (.virt 42)) (a (.wire 8 43)) := by
  have c383 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (privateBatchWrapper4_copies12 a h).1
  have c385 := (privateBatchWrapper4_copies12 a h).2.1
  have c386 := (privateBatchWrapper4_copies12 a h).2.2.1
  have c387 := (privateBatchWrapper4_copies12 a h).2.2.2.1
  have c388 := (privateBatchWrapper4_copies12 a h).2.2.2.2.1
  have e_10_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c383, ← c384, ← c385] at e_10_6
  have e_10_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c386, ← c387, ← c388] at e_10_7
  have hr := e_10_7
  simp only [e_10_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f60 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 39) = bselect (a (.wire 7 31)) (a (.virt 43)) (a (.wire 8 51)) := by
  have c389 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.1
  have c390 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.1
  have c391 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_10_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c389, ← c390, ← c391] at e_10_8
  have e_10_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c392, ← c393, ← c394] at e_10_9
  have hr := e_10_9
  simp only [e_10_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f61 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 47) = bselect (a (.wire 7 31)) (a (.virt 44)) (a (.wire 8 59)) := by
  have c395 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c395, ← c396, ← c397] at e_10_10
  have e_10_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c398, ← c399, ← c400] at e_10_11
  have hr := e_10_11
  simp only [e_10_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f62 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 55) = bselect (a (.wire 7 31)) (a (.virt 45)) (a (.wire 10 7)) := by
  have c401 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c405 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c401, ← c402, ← c403] at e_10_12
  have e_10_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c404, ← c405, ← c406] at e_10_13
  have hr := e_10_13
  simp only [e_10_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f63 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 39) = bor (a (.wire 0 35)) (a (.wire 10 11)) := by
  have c407 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_9
  simp only [← c410, ← c411, k3, ← c412] at e_0_9
  have e_9_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c407, ← c408, ← c409] at e_9_2
  have hr := e_0_9
  simp only [e_9_2] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f64 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5)) (a (.wire 10 23)) (a (.virt 116)) (a (.virt 117)) := by
  have c413 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (privateBatchWrapper4_copies13 a h).1
  have c417 := (privateBatchWrapper4_copies13 a h).2.1
  have c418 := (privateBatchWrapper4_copies13 a h).2.2.1
  have c419 := (privateBatchWrapper4_copies13 a h).2.2.2.1
  have c420 := (privateBatchWrapper4_copies13 a h).2.2.2.2.1
  have c421 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.1
  have c422 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.1
  have c423 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c419, ← c420, ← c421] at e_7_8
  have e_7_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c422, ← c423, ← c424] at e_7_9
  have e_10_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c413, k3, ← c414, k3, ← c415] at e_10_14
  have e_11_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c416, ← c417, k3, ← c418] at e_11_0
  have e_11_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c425, ← c426, k3, ← c427] at e_11_1
  refine ⟨?_, ?_⟩
  · have hc := e_7_8
    simp only [e_11_0] at hc
    linear_combination c428.trans k1 - hc
  · have hc := e_11_1
    simp only [e_10_14, e_7_9, e_11_0] at hc
    linear_combination c429.trans k1 - hc

theorem privateBatchWrapper4_f65 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 6)) (a (.wire 10 31)) (a (.virt 118)) (a (.virt 119)) := by
  have c430 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c436, ← c437, ← c438] at e_7_10
  have e_7_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c439, ← c440, ← c441] at e_7_11
  have e_11_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c430, k3, ← c431, k3, ← c432] at e_11_2
  have e_11_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c433, ← c434, k3, ← c435] at e_11_3
  have e_11_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c442, ← c443, k3, ← c444] at e_11_4
  refine ⟨?_, ?_⟩
  · have hc := e_7_10
    simp only [e_11_3] at hc
    linear_combination c445.trans k1 - hc
  · have hc := e_11_4
    simp only [e_11_2, e_7_11, e_11_3] at hc
    linear_combination c446.trans k1 - hc

theorem privateBatchWrapper4_f66 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 7)) (a (.wire 10 39)) (a (.virt 120)) (a (.virt 121)) := by
  have c447 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (privateBatchWrapper4_copies14 a h).1
  have c449 := (privateBatchWrapper4_copies14 a h).2.1
  have c450 := (privateBatchWrapper4_copies14 a h).2.2.1
  have c451 := (privateBatchWrapper4_copies14 a h).2.2.2.1
  have c452 := (privateBatchWrapper4_copies14 a h).2.2.2.2.1
  have c453 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.1
  have c454 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.1
  have c455 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c453, ← c454, ← c455] at e_7_12
  have e_7_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c456, ← c457, ← c458] at e_7_13
  have e_11_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c447, k3, ← c448, k3, ← c449] at e_11_5
  have e_11_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c450, ← c451, k3, ← c452] at e_11_6
  have e_11_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c459, ← c460, k3, ← c461] at e_11_7
  refine ⟨?_, ?_⟩
  · have hc := e_7_12
    simp only [e_11_6] at hc
    linear_combination c462.trans k1 - hc
  · have hc := e_11_7
    simp only [e_11_5, e_7_13, e_11_6] at hc
    linear_combination c463.trans k1 - hc

theorem privateBatchWrapper4_f67 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 8)) (a (.wire 10 47)) (a (.virt 122)) (a (.virt 123)) := by
  have c464 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (privateBatchWrapper4_copies15 a h).1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c470, ← c471, ← c472] at e_7_14
  have e_11_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c464, k3, ← c465, k3, ← c466] at e_11_8
  have e_11_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c467, ← c468, k3, ← c469] at e_11_9
  have e_11_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c476, ← c477, k3, ← c478] at e_11_10
  have e_12_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c473, ← c474, ← c475] at e_12_0
  refine ⟨?_, ?_⟩
  · have hc := e_7_14
    simp only [e_11_9] at hc
    linear_combination c479.trans k1 - hc
  · have hc := e_11_10
    simp only [e_11_8, e_12_0, e_11_9] at hc
    linear_combination c480.trans k1 - hc

theorem privateBatchWrapper4_f68 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 12 7) = band (a (.virt 116)) (a (.virt 118)) := by
  have c481 := (privateBatchWrapper4_copies15 a h).2.1
  have c482 := (privateBatchWrapper4_copies15 a h).2.2.1
  have c483 := (privateBatchWrapper4_copies15 a h).2.2.2.1
  have e_12_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c481, ← c482, ← c483] at e_12_1
  have hr := e_12_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f69 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 12 11) = band (a (.virt 120)) (a (.virt 122)) := by
  have c484 := (privateBatchWrapper4_copies15 a h).2.2.2.2.1
  have c485 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.1
  have c486 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.1
  have e_12_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c484, ← c485, ← c486] at e_12_2
  have hr := e_12_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f70 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 12 15) = band (a (.wire 12 7)) (a (.wire 12 11)) := by
  have c487 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have e_12_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c487, ← c488, ← c489] at e_12_3
  have hr := e_12_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f71 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 43) = bor (a (.wire 3 7)) (a (.wire 12 15)) := by
  have c490 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_10
  simp only [← c493, ← c494, k3, ← c495] at e_0_10
  have e_9_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c490, ← c491, ← c492] at e_9_3
  have hr := e_0_10
  simp only [e_9_3] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f72 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 43) = a (.virt 13) := by
  have c496 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c496

theorem privateBatchWrapper4_f74 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 19)) (a (.wire 10 23)) (a (.virt 124)) (a (.virt 125)) := by
  have c498 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (privateBatchWrapper4_copies16 a h).1
  have c513 := (privateBatchWrapper4_copies16 a h).2.1
  have c514 := (privateBatchWrapper4_copies16 a h).2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c498, k3, ← c499, k3, ← c500] at e_11_11
  have e_11_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c501, ← c502, k3, ← c503] at e_11_12
  have e_11_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c510, ← c511, k3, ← c512] at e_11_13
  have e_12_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c504, ← c505, ← c506] at e_12_4
  have e_12_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c507, ← c508, ← c509] at e_12_5
  refine ⟨?_, ?_⟩
  · have hc := e_12_4
    simp only [e_11_12] at hc
    linear_combination c513.trans k1 - hc
  · have hc := e_11_13
    simp only [e_11_11, e_12_5, e_11_12] at hc
    linear_combination c514.trans k1 - hc

theorem privateBatchWrapper4_f75 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 20)) (a (.wire 10 31)) (a (.virt 126)) (a (.virt 127)) := by
  have c515 := (privateBatchWrapper4_copies16 a h).2.2.2.1
  have c516 := (privateBatchWrapper4_copies16 a h).2.2.2.2.1
  have c517 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.1
  have c518 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.1
  have c519 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c515, k3, ← c516, k3, ← c517] at e_11_14
  have e_12_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c521, ← c522, ← c523] at e_12_6
  have e_12_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c524, ← c525, ← c526] at e_12_7
  have e_13_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c518, ← c519, k3, ← c520] at e_13_0
  have e_13_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c527, ← c528, k3, ← c529] at e_13_1
  refine ⟨?_, ?_⟩
  · have hc := e_12_6
    simp only [e_13_0] at hc
    linear_combination c530.trans k1 - hc
  · have hc := e_13_1
    simp only [e_11_14, e_12_7, e_13_0] at hc
    linear_combination c531.trans k1 - hc

theorem privateBatchWrapper4_f76 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 21)) (a (.wire 10 39)) (a (.virt 128)) (a (.virt 129)) := by
  have c532 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c535 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c541 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (privateBatchWrapper4_copies17 a h).1
  have c545 := (privateBatchWrapper4_copies17 a h).2.1
  have c546 := (privateBatchWrapper4_copies17 a h).2.2.1
  have c547 := (privateBatchWrapper4_copies17 a h).2.2.2.1
  have c548 := (privateBatchWrapper4_copies17 a h).2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c538, ← c539, ← c540] at e_12_8
  have e_12_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c541, ← c542, ← c543] at e_12_9
  have e_13_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c532, k3, ← c533, k3, ← c534] at e_13_2
  have e_13_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c535, ← c536, k3, ← c537] at e_13_3
  have e_13_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c544, ← c545, k3, ← c546] at e_13_4
  refine ⟨?_, ?_⟩
  · have hc := e_12_8
    simp only [e_13_3] at hc
    linear_combination c547.trans k1 - hc
  · have hc := e_13_4
    simp only [e_13_2, e_12_9, e_13_3] at hc
    linear_combination c548.trans k1 - hc

theorem privateBatchWrapper4_f77 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 22)) (a (.wire 10 47)) (a (.virt 130)) (a (.virt 131)) := by
  have c549 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.1
  have c550 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.1
  have c551 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_12_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c555, ← c556, ← c557] at e_12_10
  have e_12_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c558, ← c559, ← c560] at e_12_11
  have e_13_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c549, k3, ← c550, k3, ← c551] at e_13_5
  have e_13_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c552, ← c553, k3, ← c554] at e_13_6
  have e_13_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c561, ← c562, k3, ← c563] at e_13_7
  refine ⟨?_, ?_⟩
  · have hc := e_12_10
    simp only [e_13_6] at hc
    linear_combination c564.trans k1 - hc
  · have hc := e_13_7
    simp only [e_13_5, e_12_11, e_13_6] at hc
    linear_combination c565.trans k1 - hc

theorem privateBatchWrapper4_f78 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 12 51) = band (a (.virt 124)) (a (.virt 126)) := by
  have c566 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_12_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c566, ← c567, ← c568] at e_12_12
  have hr := e_12_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f79 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 12 55) = band (a (.virt 128)) (a (.virt 130)) := by
  have c569 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_12_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c569, ← c570, ← c571] at e_12_13
  have hr := e_12_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f80 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 12 59) = band (a (.wire 12 51)) (a (.wire 12 55)) := by
  have c572 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_12_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c572, ← c573, ← c574] at e_12_14
  have hr := e_12_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f81 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 47) = bor (a (.wire 3 51)) (a (.wire 12 59)) := by
  have c575 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (privateBatchWrapper4_copies18 a h).1
  have c577 := (privateBatchWrapper4_copies18 a h).2.1
  have c578 := (privateBatchWrapper4_copies18 a h).2.2.1
  have c579 := (privateBatchWrapper4_copies18 a h).2.2.2.1
  have c580 := (privateBatchWrapper4_copies18 a h).2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_11
  simp only [← c578, ← c579, k3, ← c580] at e_0_11
  have e_9_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c575, ← c576, ← c577] at e_9_4
  have hr := e_0_11
  simp only [e_9_4] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f82 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 47) = a (.virt 13) := by
  have c581 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.1
  exact c581

theorem privateBatchWrapper4_f83 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 14) = a (.virt 0) := by
  have c582 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.1
  exact c582

theorem privateBatchWrapper4_f84 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 30)) (a (.wire 10 23)) (a (.virt 132)) (a (.virt 133)) := by
  have c583 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c583, k3, ← c584, k3, ← c585] at e_13_8
  have e_13_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c586, ← c587, k3, ← c588] at e_13_9
  have e_13_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c595, ← c596, k3, ← c597] at e_13_10
  have e_14_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c589, ← c590, ← c591] at e_14_0
  have e_14_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c592, ← c593, ← c594] at e_14_1
  refine ⟨?_, ?_⟩
  · have hc := e_14_0
    simp only [e_13_9] at hc
    linear_combination c598.trans k1 - hc
  · have hc := e_13_10
    simp only [e_13_8, e_14_1, e_13_9] at hc
    linear_combination c599.trans k1 - hc

theorem privateBatchWrapper4_f85 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 31)) (a (.wire 10 31)) (a (.virt 134)) (a (.virt 135)) := by
  have c600 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (privateBatchWrapper4_copies19 a h).1
  have c609 := (privateBatchWrapper4_copies19 a h).2.1
  have c610 := (privateBatchWrapper4_copies19 a h).2.2.1
  have c611 := (privateBatchWrapper4_copies19 a h).2.2.2.1
  have c612 := (privateBatchWrapper4_copies19 a h).2.2.2.2.1
  have c613 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.1
  have c614 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.1
  have c615 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c600, k3, ← c601, k3, ← c602] at e_13_11
  have e_13_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c603, ← c604, k3, ← c605] at e_13_12
  have e_13_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c612, ← c613, k3, ← c614] at e_13_13
  have e_14_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c606, ← c607, ← c608] at e_14_2
  have e_14_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c609, ← c610, ← c611] at e_14_3
  refine ⟨?_, ?_⟩
  · have hc := e_14_2
    simp only [e_13_12] at hc
    linear_combination c615.trans k1 - hc
  · have hc := e_13_13
    simp only [e_13_11, e_14_3, e_13_12] at hc
    linear_combination c616.trans k1 - hc

theorem privateBatchWrapper4_f86 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 32)) (a (.wire 10 39)) (a (.virt 136)) (a (.virt 137)) := by
  have c617 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_13_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c617, k3, ← c618, k3, ← c619] at e_13_14
  have e_14_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c623, ← c624, ← c625] at e_14_4
  have e_14_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c626, ← c627, ← c628] at e_14_5
  have e_15_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c620, ← c621, k3, ← c622] at e_15_0
  have e_15_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c629, ← c630, k3, ← c631] at e_15_1
  refine ⟨?_, ?_⟩
  · have hc := e_14_4
    simp only [e_15_0] at hc
    linear_combination c632.trans k1 - hc
  · have hc := e_15_1
    simp only [e_13_14, e_14_5, e_15_0] at hc
    linear_combination c633.trans k1 - hc

theorem privateBatchWrapper4_f87 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 33)) (a (.wire 10 47)) (a (.virt 138)) (a (.virt 139)) := by
  have c634 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (privateBatchWrapper4_copies20 a h).1
  have c641 := (privateBatchWrapper4_copies20 a h).2.1
  have c642 := (privateBatchWrapper4_copies20 a h).2.2.1
  have c643 := (privateBatchWrapper4_copies20 a h).2.2.2.1
  have c644 := (privateBatchWrapper4_copies20 a h).2.2.2.2.1
  have c645 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.1
  have c646 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.1
  have c647 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c640, ← c641, ← c642] at e_14_6
  have e_14_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c643, ← c644, ← c645] at e_14_7
  have e_15_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c634, k3, ← c635, k3, ← c636] at e_15_2
  have e_15_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c637, ← c638, k3, ← c639] at e_15_3
  have e_15_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c646, ← c647, k3, ← c648] at e_15_4
  refine ⟨?_, ?_⟩
  · have hc := e_14_6
    simp only [e_15_3] at hc
    linear_combination c649.trans k1 - hc
  · have hc := e_15_4
    simp only [e_15_2, e_14_7, e_15_3] at hc
    linear_combination c650.trans k1 - hc

theorem privateBatchWrapper4_f88 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 14 35) = band (a (.virt 132)) (a (.virt 134)) := by
  have c651 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c651, ← c652, ← c653] at e_14_8
  have hr := e_14_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f89 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 14 39) = band (a (.virt 136)) (a (.virt 138)) := by
  have c654 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c654, ← c655, ← c656] at e_14_9
  have hr := e_14_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f90 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 14 43) = band (a (.wire 14 35)) (a (.wire 14 39)) := by
  have c657 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c657, ← c658, ← c659] at e_14_10
  have hr := e_14_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f91 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 51) = bor (a (.wire 5 35)) (a (.wire 14 43)) := by
  have c660 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c663, ← c664, k3, ← c665] at e_0_12
  have e_9_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c660, ← c661, ← c662] at e_9_5
  have hr := e_0_12
  simp only [e_9_5] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f92 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 51) = a (.virt 13) := by
  have c666 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c666

theorem privateBatchWrapper4_f93 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 25) = a (.virt 0) := by
  have c667 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c667

theorem privateBatchWrapper4_f94 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 41)) (a (.wire 10 23)) (a (.virt 140)) (a (.virt 141)) := by
  have c668 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (privateBatchWrapper4_copies21 a h).1
  have c673 := (privateBatchWrapper4_copies21 a h).2.1
  have c674 := (privateBatchWrapper4_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper4_copies21 a h).2.2.2.1
  have c676 := (privateBatchWrapper4_copies21 a h).2.2.2.2.1
  have c677 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.1
  have c678 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.1
  have c679 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c674, ← c675, ← c676] at e_14_11
  have e_14_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c677, ← c678, ← c679] at e_14_12
  have e_15_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c668, k3, ← c669, k3, ← c670] at e_15_5
  have e_15_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c671, ← c672, k3, ← c673] at e_15_6
  have e_15_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_7
  simp only [← c680, ← c681, k3, ← c682] at e_15_7
  refine ⟨?_, ?_⟩
  · have hc := e_14_11
    simp only [e_15_6] at hc
    linear_combination c683.trans k1 - hc
  · have hc := e_15_7
    simp only [e_15_5, e_14_12, e_15_6] at hc
    linear_combination c684.trans k1 - hc

theorem privateBatchWrapper4_f95 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 42)) (a (.wire 10 31)) (a (.virt 142)) (a (.virt 143)) := by
  have c685 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_14_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c691, ← c692, ← c693] at e_14_13
  have e_14_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c694, ← c695, ← c696] at e_14_14
  have e_15_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_8
  simp only [← c685, k3, ← c686, k3, ← c687] at e_15_8
  have e_15_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_9
  simp only [← c688, ← c689, k3, ← c690] at e_15_9
  have e_15_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_10
  simp only [← c697, ← c698, k3, ← c699] at e_15_10
  refine ⟨?_, ?_⟩
  · have hc := e_14_13
    simp only [e_15_9] at hc
    linear_combination c700.trans k1 - hc
  · have hc := e_15_10
    simp only [e_15_8, e_14_14, e_15_9] at hc
    linear_combination c701.trans k1 - hc

theorem privateBatchWrapper4_f96 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 43)) (a (.wire 10 39)) (a (.virt 144)) (a (.virt 145)) := by
  have c702 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (privateBatchWrapper4_copies22 a h).1
  have c705 := (privateBatchWrapper4_copies22 a h).2.1
  have c706 := (privateBatchWrapper4_copies22 a h).2.2.1
  have c707 := (privateBatchWrapper4_copies22 a h).2.2.2.1
  have c708 := (privateBatchWrapper4_copies22 a h).2.2.2.2.1
  have c709 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.1
  have c710 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.1
  have c711 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_11
  simp only [← c702, k3, ← c703, k3, ← c704] at e_15_11
  have e_15_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_12
  simp only [← c705, ← c706, k3, ← c707] at e_15_12
  have e_15_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_13
  simp only [← c714, ← c715, k3, ← c716] at e_15_13
  have e_16_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c708, ← c709, ← c710] at e_16_0
  have e_16_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c711, ← c712, ← c713] at e_16_1
  refine ⟨?_, ?_⟩
  · have hc := e_16_0
    simp only [e_15_12] at hc
    linear_combination c717.trans k1 - hc
  · have hc := e_15_13
    simp only [e_15_11, e_16_1, e_15_12] at hc
    linear_combination c718.trans k1 - hc

theorem privateBatchWrapper4_f97 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 44)) (a (.wire 10 47)) (a (.virt 146)) (a (.virt 147)) := by
  have c719 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c733 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_15_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_14
  simp only [← c719, k3, ← c720, k3, ← c721] at e_15_14
  have e_16_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c725, ← c726, ← c727] at e_16_2
  have e_16_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c728, ← c729, ← c730] at e_16_3
  have e_17_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c722, ← c723, k3, ← c724] at e_17_0
  have e_17_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c731, ← c732, k3, ← c733] at e_17_1
  refine ⟨?_, ?_⟩
  · have hc := e_16_2
    simp only [e_17_0] at hc
    linear_combination c734.trans k1 - hc
  · have hc := e_17_1
    simp only [e_15_14, e_16_3, e_17_0] at hc
    linear_combination c735.trans k1 - hc

theorem privateBatchWrapper4_f98 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 19) = band (a (.virt 140)) (a (.virt 142)) := by
  have c736 := (privateBatchWrapper4_copies23 a h).1
  have c737 := (privateBatchWrapper4_copies23 a h).2.1
  have c738 := (privateBatchWrapper4_copies23 a h).2.2.1
  have e_16_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c736, ← c737, ← c738] at e_16_4
  have hr := e_16_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f99 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 23) = band (a (.virt 144)) (a (.virt 146)) := by
  have c739 := (privateBatchWrapper4_copies23 a h).2.2.2.1
  have c740 := (privateBatchWrapper4_copies23 a h).2.2.2.2.1
  have c741 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.1
  have e_16_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c739, ← c740, ← c741] at e_16_5
  have hr := e_16_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f100 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 27) = band (a (.wire 16 19)) (a (.wire 16 23)) := by
  have c742 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.1
  have c743 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.1
  have e_16_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c742, ← c743, ← c744] at e_16_6
  have hr := e_16_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f101 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 55) = bor (a (.wire 7 19)) (a (.wire 16 27)) := by
  have c745 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c748, ← c749, k3, ← c750] at e_0_13
  have e_9_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c745, ← c746, ← c747] at e_9_6
  have hr := e_0_13
  simp only [e_9_6] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f102 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 55) = a (.virt 13) := by
  have c751 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c751

theorem privateBatchWrapper4_f103 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 36) = a (.virt 0) := by
  have c752 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c752

theorem privateBatchWrapper4_f104 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 15) = bselect (a (.wire 3 7)) (a (.virt 11)) (a (.virt 10)) := by
  have c753 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c753, ← c754, ← c755] at e_17_2
  have e_17_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c756, ← c757, k1, ← c758] at e_17_3
  have hr := e_17_3
  simp only [e_17_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f105 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 15) = a (.virt 11) + a (.wire 17 15) := by
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [k1]
  ring

theorem privateBatchWrapper4_f106 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 23) = bselect (a (.wire 3 51)) (a (.virt 11)) (a (.virt 24)) := by
  have c759 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c759, ← c760, ← c761] at e_17_4
  have e_17_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c762, ← c763, k1, ← c764] at e_17_5
  have hr := e_17_5
  simp only [e_17_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f107 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 0 59) = a (.wire 17 15) + a (.wire 17 23) := by
  have c765 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c765, ← c766, k3, ← c767] at e_0_14
  have hr := e_0_14
  linear_combination hr

theorem privateBatchWrapper4_f108 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 31) = bselect (a (.wire 5 35)) (a (.virt 11)) (a (.virt 35)) := by
  have c768 := (privateBatchWrapper4_copies24 a h).1
  have c769 := (privateBatchWrapper4_copies24 a h).2.1
  have c770 := (privateBatchWrapper4_copies24 a h).2.2.1
  have c771 := (privateBatchWrapper4_copies24 a h).2.2.2.1
  have c772 := (privateBatchWrapper4_copies24 a h).2.2.2.2.1
  have c773 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c768, ← c769, ← c770] at e_17_6
  have e_17_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c771, ← c772, k1, ← c773] at e_17_7
  have hr := e_17_7
  simp only [e_17_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f109 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 18 3) = a (.wire 0 59) + a (.wire 17 31) := by
  have c774 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.1
  have c775 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c774, ← c775, k3, ← c776] at e_18_0
  have hr := e_18_0
  linear_combination hr

theorem privateBatchWrapper4_f110 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 39) = bselect (a (.wire 7 19)) (a (.virt 11)) (a (.virt 46)) := by
  have c777 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have c778 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c781 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c777, ← c778, ← c779] at e_17_8
  have e_17_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c780, ← c781, k1, ← c782] at e_17_9
  have hr := e_17_9
  simp only [e_17_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f111 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 18 7) = a (.wire 18 3) + a (.wire 17 39) := by
  have c783 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c783, ← c784, k3, ← c785] at e_18_1
  have hr := e_18_1
  linear_combination hr

theorem privateBatchWrapper4_f112 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 52)) 32 := by
  have c786 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (privateBatchWrapper4_copies25 a h).1
  have c801 := (privateBatchWrapper4_copies25 a h).2.1
  have c802 := (privateBatchWrapper4_copies25 a h).2.2.1
  have c803 := (privateBatchWrapper4_copies25 a h).2.2.2.1
  have c804 := (privateBatchWrapper4_copies25 a h).2.2.2.2.1
  have c805 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.1
  have c806 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.1
  have c807 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 19) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c786.trans k1
    · exact c787.trans k1
    · exact c788.trans k1
    · exact c789.trans k1
    · exact c790.trans k1
    · exact c791.trans k1
    · exact c792.trans k1
    · exact c793.trans k1
    · exact c794.trans k1
    · exact c795.trans k1
    · exact c796.trans k1
    · exact c797.trans k1
    · exact c798.trans k1
    · exact c799.trans k1
    · exact c800.trans k1
    · exact c801.trans k1
    · exact c802.trans k1
    · exact c803.trans k1
    · exact c804.trans k1
    · exact c805.trans k1
    · exact c806.trans k1
    · exact c807.trans k1
    · exact c808.trans k1
    · exact c809.trans k1
    · exact c810.trans k1
    · exact c811.trans k1
    · exact c812.trans k1
    )
  rwa [c813] at hr

theorem privateBatchWrapper4_f113 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 52) = a (.virt 11) + a (.virt 52) := by
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [k1]
  ring

theorem privateBatchWrapper4_f114 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 57)) 32 := by
  have c814 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c829 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c832 := (privateBatchWrapper4_copies26 a h).1
  have c833 := (privateBatchWrapper4_copies26 a h).2.1
  have c834 := (privateBatchWrapper4_copies26 a h).2.2.1
  have c835 := (privateBatchWrapper4_copies26 a h).2.2.2.1
  have c836 := (privateBatchWrapper4_copies26 a h).2.2.2.2.1
  have c837 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.1
  have c838 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.1
  have c839 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.1
  have c841 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 20) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c814.trans k1
    · exact c815.trans k1
    · exact c816.trans k1
    · exact c817.trans k1
    · exact c818.trans k1
    · exact c819.trans k1
    · exact c820.trans k1
    · exact c821.trans k1
    · exact c822.trans k1
    · exact c823.trans k1
    · exact c824.trans k1
    · exact c825.trans k1
    · exact c826.trans k1
    · exact c827.trans k1
    · exact c828.trans k1
    · exact c829.trans k1
    · exact c830.trans k1
    · exact c831.trans k1
    · exact c832.trans k1
    · exact c833.trans k1
    · exact c834.trans k1
    · exact c835.trans k1
    · exact c836.trans k1
    · exact c837.trans k1
    · exact c838.trans k1
    · exact c839.trans k1
    · exact c840.trans k1
    )
  rwa [c841] at hr

theorem privateBatchWrapper4_f115 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 18 11) = a (.virt 52) + a (.virt 57) := by
  have c842 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c844 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c842, ← c843, k3, ← c844] at e_18_2
  have hr := e_18_2
  linear_combination hr

theorem privateBatchWrapper4_f116 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 62)) 32 := by
  have c845 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c847 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c853 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c860 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c863 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (privateBatchWrapper4_copies27 a h).1
  have c865 := (privateBatchWrapper4_copies27 a h).2.1
  have c866 := (privateBatchWrapper4_copies27 a h).2.2.1
  have c867 := (privateBatchWrapper4_copies27 a h).2.2.2.1
  have c868 := (privateBatchWrapper4_copies27 a h).2.2.2.2.1
  have c869 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.1
  have c870 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.1
  have c871 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 21) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c845.trans k1
    · exact c846.trans k1
    · exact c847.trans k1
    · exact c848.trans k1
    · exact c849.trans k1
    · exact c850.trans k1
    · exact c851.trans k1
    · exact c852.trans k1
    · exact c853.trans k1
    · exact c854.trans k1
    · exact c855.trans k1
    · exact c856.trans k1
    · exact c857.trans k1
    · exact c858.trans k1
    · exact c859.trans k1
    · exact c860.trans k1
    · exact c861.trans k1
    · exact c862.trans k1
    · exact c863.trans k1
    · exact c864.trans k1
    · exact c865.trans k1
    · exact c866.trans k1
    · exact c867.trans k1
    · exact c868.trans k1
    · exact c869.trans k1
    · exact c870.trans k1
    · exact c871.trans k1
    )
  rwa [c872] at hr

theorem privateBatchWrapper4_f117 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 18 15) = a (.wire 18 11) + a (.virt 62) := by
  have c873 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c873, ← c874, k3, ← c875] at e_18_3
  have hr := e_18_3
  linear_combination hr

theorem privateBatchWrapper4_f118 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 67)) 32 := by
  have c876 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c880 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c883 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (privateBatchWrapper4_copies28 a h).1
  have c897 := (privateBatchWrapper4_copies28 a h).2.1
  have c898 := (privateBatchWrapper4_copies28 a h).2.2.1
  have c899 := (privateBatchWrapper4_copies28 a h).2.2.2.1
  have c900 := (privateBatchWrapper4_copies28 a h).2.2.2.2.1
  have c901 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.1
  have c902 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.1
  have c903 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 22) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c876.trans k1
    · exact c877.trans k1
    · exact c878.trans k1
    · exact c879.trans k1
    · exact c880.trans k1
    · exact c881.trans k1
    · exact c882.trans k1
    · exact c883.trans k1
    · exact c884.trans k1
    · exact c885.trans k1
    · exact c886.trans k1
    · exact c887.trans k1
    · exact c888.trans k1
    · exact c889.trans k1
    · exact c890.trans k1
    · exact c891.trans k1
    · exact c892.trans k1
    · exact c893.trans k1
    · exact c894.trans k1
    · exact c895.trans k1
    · exact c896.trans k1
    · exact c897.trans k1
    · exact c898.trans k1
    · exact c899.trans k1
    · exact c900.trans k1
    · exact c901.trans k1
    · exact c902.trans k1
    )
  rwa [c903] at hr

theorem privateBatchWrapper4_f119 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 18 19) = a (.wire 18 15) + a (.virt 67) := by
  have c904 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_18_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c904, ← c905, k3, ← c906] at e_18_4
  have hr := e_18_4
  linear_combination hr

theorem privateBatchWrapper4_f120 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 47)) 14 := by
  have c907 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c925 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c927 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c928 := (privateBatchWrapper4_copies29 a h).1
  have c929 := (privateBatchWrapper4_copies29 a h).2.1
  have c930 := (privateBatchWrapper4_copies29 a h).2.2.1
  have c931 := (privateBatchWrapper4_copies29 a h).2.2.2.1
  have c932 := (privateBatchWrapper4_copies29 a h).2.2.2.2.1
  have c933 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.1
  have c934 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.1
  have c935 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.1
  have c937 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c947 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 23) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c907.trans k1
    · exact c908.trans k1
    · exact c909.trans k1
    · exact c910.trans k1
    · exact c911.trans k1
    · exact c912.trans k1
    · exact c913.trans k1
    · exact c914.trans k1
    · exact c915.trans k1
    · exact c916.trans k1
    · exact c917.trans k1
    · exact c918.trans k1
    · exact c919.trans k1
    · exact c920.trans k1
    · exact c921.trans k1
    · exact c922.trans k1
    · exact c923.trans k1
    · exact c924.trans k1
    · exact c925.trans k1
    · exact c926.trans k1
    · exact c927.trans k1
    · exact c928.trans k1
    · exact c929.trans k1
    · exact c930.trans k1
    · exact c931.trans k1
    · exact c932.trans k1
    · exact c933.trans k1
    · exact c934.trans k1
    · exact c935.trans k1
    · exact c936.trans k1
    · exact c937.trans k1
    · exact c938.trans k1
    · exact c939.trans k1
    · exact c940.trans k1
    · exact c941.trans k1
    · exact c942.trans k1
    · exact c943.trans k1
    · exact c944.trans k1
    · exact c945.trans k1
    · exact c946.trans k1
    · exact c947.trans k1
    · exact c948.trans k1
    · exact c949.trans k1
    · exact c950.trans k1
    · exact c951.trans k1
    )
  rwa [c952] at hr

theorem privateBatchWrapper4_f121 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 43) = a (.virt 148) - a (.virt 47) := by
  have c953 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c953, k0, ← c954, k3, ← c955] at e_17_10
  have hr := e_17_10
  simp only [k0]
  linear_combination hr

theorem privateBatchWrapper4_f122 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.wire 17 43)) 14 := by
  have c956 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c959 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (privateBatchWrapper4_copies30 a h).1
  have c961 := (privateBatchWrapper4_copies30 a h).2.1
  have c962 := (privateBatchWrapper4_copies30 a h).2.2.1
  have c963 := (privateBatchWrapper4_copies30 a h).2.2.2.1
  have c964 := (privateBatchWrapper4_copies30 a h).2.2.2.2.1
  have c965 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.1
  have c966 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.1
  have c967 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  have c971 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c977 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c987 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c989 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c990 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (privateBatchWrapper4_copies31 a h).1
  have c993 := (privateBatchWrapper4_copies31 a h).2.1
  have c994 := (privateBatchWrapper4_copies31 a h).2.2.1
  have c995 := (privateBatchWrapper4_copies31 a h).2.2.2.1
  have c996 := (privateBatchWrapper4_copies31 a h).2.2.2.2.1
  have c997 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.1
  have c998 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.1
  have c999 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 24) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c956.trans k1
    · exact c957.trans k1
    · exact c958.trans k1
    · exact c959.trans k1
    · exact c960.trans k1
    · exact c961.trans k1
    · exact c962.trans k1
    · exact c963.trans k1
    · exact c964.trans k1
    · exact c965.trans k1
    · exact c966.trans k1
    · exact c967.trans k1
    · exact c968.trans k1
    · exact c969.trans k1
    · exact c970.trans k1
    · exact c971.trans k1
    · exact c972.trans k1
    · exact c973.trans k1
    · exact c974.trans k1
    · exact c975.trans k1
    · exact c976.trans k1
    · exact c977.trans k1
    · exact c978.trans k1
    · exact c979.trans k1
    · exact c980.trans k1
    · exact c981.trans k1
    · exact c982.trans k1
    · exact c983.trans k1
    · exact c984.trans k1
    · exact c985.trans k1
    · exact c986.trans k1
    · exact c987.trans k1
    · exact c988.trans k1
    · exact c989.trans k1
    · exact c990.trans k1
    · exact c991.trans k1
    · exact c992.trans k1
    · exact c993.trans k1
    · exact c994.trans k1
    · exact c995.trans k1
    · exact c996.trans k1
    · exact c997.trans k1
    · exact c998.trans k1
    · exact c999.trans k1
    · exact c1000.trans k1
    )
  rwa [c1001] at hr

theorem privateBatchWrapper4_f123 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 31) = a (.wire 18 19) * a (.virt 148) := by
  have c1002 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_16_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c1002, ← c1003, k0, ← c1004] at e_16_7
  have hr := e_16_7
  simp only [k0]
  linear_combination hr

theorem privateBatchWrapper4_f124 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 35) = a (.wire 18 7) * a (.wire 17 43) := by
  have c1005 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c1005, ← c1006, ← c1007] at e_16_8
  have hr := e_16_8
  linear_combination hr

theorem privateBatchWrapper4_f125 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 47) = a (.wire 16 35) - a (.wire 16 31) := by
  have c1008 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_17_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c1008, ← c1009, k3, ← c1010] at e_17_11
  have hr := e_17_11
  linear_combination hr

theorem privateBatchWrapper4_f126 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.wire 17 47)) 52 := by
  have c1011 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1012 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1017 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 25) (N := 59) (n := 52) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1011.trans k1
    · exact c1012.trans k1
    · exact c1013.trans k1
    · exact c1014.trans k1
    · exact c1015.trans k1
    · exact c1016.trans k1
    · exact c1017.trans k1
    )
  rwa [c1018] at hr

theorem privateBatchWrapper4_f127 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 11) = bnot (a (.wire 3 7)) := by
  have c260 := (privateBatchWrapper4_copies8 a h).2.2.2.2.1
  have c261 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c260, k3, ← c261, k3, ← c262] at e_6_2
  have hr := e_6_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f128 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 35) = bnot (a (.wire 3 51)) := by
  have c278 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c278, k3, ← c279, k3, ← c280] at e_6_8
  have hr := e_6_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f129 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 39) = band (a (.wire 6 11)) (a (.wire 6 35)) := by
  have c1019 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c1019, ← c1020, ← c1021] at e_16_9
  have hr := e_16_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f130 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 15)) (a (.virt 149)) (a (.virt 150)) := by
  have c1022 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1023 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1024 := (privateBatchWrapper4_copies32 a h).1
  have c1025 := (privateBatchWrapper4_copies32 a h).2.1
  have c1026 := (privateBatchWrapper4_copies32 a h).2.2.1
  have c1027 := (privateBatchWrapper4_copies32 a h).2.2.2.1
  have c1028 := (privateBatchWrapper4_copies32 a h).2.2.2.2.1
  have c1029 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.1
  have c1030 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_16_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c1028, ← c1029, ← c1030] at e_16_10
  have e_16_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1031, ← c1032, ← c1033] at e_16_11
  have e_17_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c1022, k3, ← c1023, k3, ← c1024] at e_17_12
  have e_17_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c1025, ← c1026, k3, ← c1027] at e_17_13
  have e_17_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c1034, ← c1035, k3, ← c1036] at e_17_14
  refine ⟨?_, ?_⟩
  · have hc := e_16_10
    simp only [e_17_13] at hc
    linear_combination c1037.trans k1 - hc
  · have hc := e_17_14
    simp only [e_17_12, e_16_11, e_17_13] at hc
    linear_combination c1038.trans k1 - hc

theorem privateBatchWrapper4_f131 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 16)) (a (.virt 151)) (a (.virt 152)) := by
  have c1039 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1041 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_16_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1045, ← c1046, ← c1047] at e_16_12
  have e_16_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1048, ← c1049, ← c1050] at e_16_13
  have e_26_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1039, k3, ← c1040, k3, ← c1041] at e_26_0
  have e_26_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1042, ← c1043, k3, ← c1044] at e_26_1
  have e_26_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1051, ← c1052, k3, ← c1053] at e_26_2
  refine ⟨?_, ?_⟩
  · have hc := e_16_12
    simp only [e_26_1] at hc
    linear_combination c1054.trans k1 - hc
  · have hc := e_26_2
    simp only [e_26_0, e_16_13, e_26_1] at hc
    linear_combination c1055.trans k1 - hc

theorem privateBatchWrapper4_f132 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 17)) (a (.virt 153)) (a (.virt 154)) := by
  have c1056 := (privateBatchWrapper4_copies33 a h).1
  have c1057 := (privateBatchWrapper4_copies33 a h).2.1
  have c1058 := (privateBatchWrapper4_copies33 a h).2.2.1
  have c1059 := (privateBatchWrapper4_copies33 a h).2.2.2.1
  have c1060 := (privateBatchWrapper4_copies33 a h).2.2.2.2.1
  have c1061 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.1
  have c1062 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_16_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1062, ← c1063, ← c1064] at e_16_14
  have e_26_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1056, k3, ← c1057, k3, ← c1058] at e_26_3
  have e_26_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1059, ← c1060, k3, ← c1061] at e_26_4
  have e_26_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1068, ← c1069, k3, ← c1070] at e_26_5
  have e_27_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1065, ← c1066, ← c1067] at e_27_0
  refine ⟨?_, ?_⟩
  · have hc := e_16_14
    simp only [e_26_4] at hc
    linear_combination c1071.trans k1 - hc
  · have hc := e_26_5
    simp only [e_26_3, e_27_0, e_26_4] at hc
    linear_combination c1072.trans k1 - hc

theorem privateBatchWrapper4_f133 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 18)) (a (.virt 155)) (a (.virt 156)) := by
  have c1073 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1078 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (privateBatchWrapper4_copies34 a h).1
  have c1089 := (privateBatchWrapper4_copies34 a h).2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_26_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1073, k3, ← c1074, k3, ← c1075] at e_26_6
  have e_26_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1076, ← c1077, k3, ← c1078] at e_26_7
  have e_26_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1085, ← c1086, k3, ← c1087] at e_26_8
  have e_27_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1079, ← c1080, ← c1081] at e_27_1
  have e_27_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1082, ← c1083, ← c1084] at e_27_2
  refine ⟨?_, ?_⟩
  · have hc := e_27_1
    simp only [e_26_7] at hc
    linear_combination c1088.trans k1 - hc
  · have hc := e_26_8
    simp only [e_26_6, e_27_2, e_26_7] at hc
    linear_combination c1089.trans k1 - hc

theorem privateBatchWrapper4_f134 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 15) = band (a (.virt 149)) (a (.virt 151)) := by
  have c1090 := (privateBatchWrapper4_copies34 a h).2.2.1
  have c1091 := (privateBatchWrapper4_copies34 a h).2.2.2.1
  have c1092 := (privateBatchWrapper4_copies34 a h).2.2.2.2.1
  have e_27_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1090, ← c1091, ← c1092] at e_27_3
  have hr := e_27_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f135 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 19) = band (a (.virt 153)) (a (.virt 155)) := by
  have c1093 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.1
  have c1094 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.1
  have e_27_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_4
  simp only [← c1093, ← c1094, ← c1095] at e_27_4
  have hr := e_27_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f136 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 23) = band (a (.wire 27 15)) (a (.wire 27 19)) := by
  have c1096 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_27_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_5
  simp only [← c1096, ← c1097, ← c1098] at e_27_5
  have hr := e_27_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f137 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 27) = band (a (.wire 16 39)) (a (.wire 27 23)) := by
  have c1099 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_6
  simp only [← c1099, ← c1100, ← c1101] at e_27_6
  have hr := e_27_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f138 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 27) = a (.virt 11) := by
  have c1102 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1102

theorem privateBatchWrapper4_f139 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 23) = bnot (a (.wire 5 35)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c323, k3, ← c324, k3, ← c325] at e_8_5
  have hr := e_8_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f140 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 31) = band (a (.wire 6 11)) (a (.wire 8 23)) := by
  have c1103 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_7
  simp only [← c1103, ← c1104, ← c1105] at e_27_7
  have hr := e_27_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f141 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 26)) (a (.virt 157)) (a (.virt 158)) := by
  have c1106 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1107 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1108 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1120 := (privateBatchWrapper4_copies35 a h).1
  have c1121 := (privateBatchWrapper4_copies35 a h).2.1
  have c1122 := (privateBatchWrapper4_copies35 a h).2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_26_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1106, k3, ← c1107, k3, ← c1108] at e_26_9
  have e_26_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1109, ← c1110, k3, ← c1111] at e_26_10
  have e_26_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1118, ← c1119, k3, ← c1120] at e_26_11
  have e_27_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_8
  simp only [← c1112, ← c1113, ← c1114] at e_27_8
  have e_27_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_9
  simp only [← c1115, ← c1116, ← c1117] at e_27_9
  refine ⟨?_, ?_⟩
  · have hc := e_27_8
    simp only [e_26_10] at hc
    linear_combination c1121.trans k1 - hc
  · have hc := e_26_11
    simp only [e_26_9, e_27_9, e_26_10] at hc
    linear_combination c1122.trans k1 - hc

theorem privateBatchWrapper4_f142 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 27)) (a (.virt 159)) (a (.virt 160)) := by
  have c1123 := (privateBatchWrapper4_copies35 a h).2.2.2.1
  have c1124 := (privateBatchWrapper4_copies35 a h).2.2.2.2.1
  have c1125 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.1
  have c1126 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_26_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1123, k3, ← c1124, k3, ← c1125] at e_26_12
  have e_26_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1126, ← c1127, k3, ← c1128] at e_26_13
  have e_26_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1135, ← c1136, k3, ← c1137] at e_26_14
  have e_27_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_10
  simp only [← c1129, ← c1130, ← c1131] at e_27_10
  have e_27_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_11
  simp only [← c1132, ← c1133, ← c1134] at e_27_11
  refine ⟨?_, ?_⟩
  · have hc := e_27_10
    simp only [e_26_13] at hc
    linear_combination c1138.trans k1 - hc
  · have hc := e_26_14
    simp only [e_26_12, e_27_11, e_26_13] at hc
    linear_combination c1139.trans k1 - hc

theorem privateBatchWrapper4_f143 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 28)) (a (.virt 161)) (a (.virt 162)) := by
  have c1140 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1143 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1150 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (privateBatchWrapper4_copies36 a h).1
  have c1153 := (privateBatchWrapper4_copies36 a h).2.1
  have c1154 := (privateBatchWrapper4_copies36 a h).2.2.1
  have c1155 := (privateBatchWrapper4_copies36 a h).2.2.2.1
  have c1156 := (privateBatchWrapper4_copies36 a h).2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_27_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_12
  simp only [← c1146, ← c1147, ← c1148] at e_27_12
  have e_27_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_13
  simp only [← c1149, ← c1150, ← c1151] at e_27_13
  have e_28_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c1140, k3, ← c1141, k3, ← c1142] at e_28_0
  have e_28_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c1143, ← c1144, k3, ← c1145] at e_28_1
  have e_28_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c1152, ← c1153, k3, ← c1154] at e_28_2
  refine ⟨?_, ?_⟩
  · have hc := e_27_12
    simp only [e_28_1] at hc
    linear_combination c1155.trans k1 - hc
  · have hc := e_28_2
    simp only [e_28_0, e_27_13, e_28_1] at hc
    linear_combination c1156.trans k1 - hc

theorem privateBatchWrapper4_f144 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 29)) (a (.virt 163)) (a (.virt 164)) := by
  have c1157 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.1
  have c1158 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.1
  have c1159 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have c1162 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_27_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_14
  simp only [← c1163, ← c1164, ← c1165] at e_27_14
  have e_28_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c1157, k3, ← c1158, k3, ← c1159] at e_28_3
  have e_28_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c1160, ← c1161, k3, ← c1162] at e_28_4
  have e_28_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1169, ← c1170, k3, ← c1171] at e_28_5
  have e_29_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1166, ← c1167, ← c1168] at e_29_0
  refine ⟨?_, ?_⟩
  · have hc := e_27_14
    simp only [e_28_4] at hc
    linear_combination c1172.trans k1 - hc
  · have hc := e_28_5
    simp only [e_28_3, e_29_0, e_28_4] at hc
    linear_combination c1173.trans k1 - hc

theorem privateBatchWrapper4_f145 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 7) = band (a (.virt 157)) (a (.virt 159)) := by
  have c1174 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1174, ← c1175, ← c1176] at e_29_1
  have hr := e_29_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f146 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 11) = band (a (.virt 161)) (a (.virt 163)) := by
  have c1177 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1179 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1177, ← c1178, ← c1179] at e_29_2
  have hr := e_29_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f147 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 15) = band (a (.wire 29 7)) (a (.wire 29 11)) := by
  have c1180 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1182 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1180, ← c1181, ← c1182] at e_29_3
  have hr := e_29_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f148 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 19) = band (a (.wire 27 31)) (a (.wire 29 15)) := by
  have c1183 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (privateBatchWrapper4_copies37 a h).1
  have c1185 := (privateBatchWrapper4_copies37 a h).2.1
  have e_29_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1183, ← c1184, ← c1185] at e_29_4
  have hr := e_29_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f149 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 19) = a (.virt 11) := by
  have c1186 := (privateBatchWrapper4_copies37 a h).2.2.1
  exact c1186

theorem privateBatchWrapper4_f150 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 7 19)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k3, ← c369, k3, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f151 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 23) = band (a (.wire 6 11)) (a (.wire 10 11)) := by
  have c1187 := (privateBatchWrapper4_copies37 a h).2.2.2.1
  have c1188 := (privateBatchWrapper4_copies37 a h).2.2.2.2.1
  have c1189 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.1
  have e_29_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1187, ← c1188, ← c1189] at e_29_5
  have hr := e_29_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f152 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 37)) (a (.virt 165)) (a (.virt 166)) := by
  have c1190 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.1
  have c1191 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_28_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1190, k3, ← c1191, k3, ← c1192] at e_28_6
  have e_28_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1193, ← c1194, k3, ← c1195] at e_28_7
  have e_28_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1202, ← c1203, k3, ← c1204] at e_28_8
  have e_29_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1196, ← c1197, ← c1198] at e_29_6
  have e_29_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c1199, ← c1200, ← c1201] at e_29_7
  refine ⟨?_, ?_⟩
  · have hc := e_29_6
    simp only [e_28_7] at hc
    linear_combination c1205.trans k1 - hc
  · have hc := e_28_8
    simp only [e_28_6, e_29_7, e_28_7] at hc
    linear_combination c1206.trans k1 - hc

theorem privateBatchWrapper4_f153 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 38)) (a (.virt 167)) (a (.virt 168)) := by
  have c1207 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (privateBatchWrapper4_copies38 a h).1
  have c1217 := (privateBatchWrapper4_copies38 a h).2.1
  have c1218 := (privateBatchWrapper4_copies38 a h).2.2.1
  have c1219 := (privateBatchWrapper4_copies38 a h).2.2.2.1
  have c1220 := (privateBatchWrapper4_copies38 a h).2.2.2.2.1
  have c1221 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.1
  have c1222 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_28_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1207, k3, ← c1208, k3, ← c1209] at e_28_9
  have e_28_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1210, ← c1211, k3, ← c1212] at e_28_10
  have e_28_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1219, ← c1220, k3, ← c1221] at e_28_11
  have e_29_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c1213, ← c1214, ← c1215] at e_29_8
  have e_29_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c1216, ← c1217, ← c1218] at e_29_9
  refine ⟨?_, ?_⟩
  · have hc := e_29_8
    simp only [e_28_10] at hc
    linear_combination c1222.trans k1 - hc
  · have hc := e_28_11
    simp only [e_28_9, e_29_9, e_28_10] at hc
    linear_combination c1223.trans k1 - hc

theorem privateBatchWrapper4_f154 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 39)) (a (.virt 169)) (a (.virt 170)) := by
  have c1224 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1227 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1233 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1236 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1239 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_28_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1224, k3, ← c1225, k3, ← c1226] at e_28_12
  have e_28_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1227, ← c1228, k3, ← c1229] at e_28_13
  have e_28_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1236, ← c1237, k3, ← c1238] at e_28_14
  have e_29_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c1230, ← c1231, ← c1232] at e_29_10
  have e_29_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c1233, ← c1234, ← c1235] at e_29_11
  refine ⟨?_, ?_⟩
  · have hc := e_29_10
    simp only [e_28_13] at hc
    linear_combination c1239.trans k1 - hc
  · have hc := e_28_14
    simp only [e_28_12, e_29_11, e_28_13] at hc
    linear_combination c1240.trans k1 - hc

theorem privateBatchWrapper4_f155 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 40)) (a (.virt 171)) (a (.virt 172)) := by
  have c1241 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1245 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (privateBatchWrapper4_copies39 a h).1
  have c1249 := (privateBatchWrapper4_copies39 a h).2.1
  have c1250 := (privateBatchWrapper4_copies39 a h).2.2.1
  have c1251 := (privateBatchWrapper4_copies39 a h).2.2.2.1
  have c1252 := (privateBatchWrapper4_copies39 a h).2.2.2.2.1
  have c1253 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.1
  have c1254 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.1
  have c1256 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_29_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c1247, ← c1248, ← c1249] at e_29_12
  have e_29_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c1250, ← c1251, ← c1252] at e_29_13
  have e_30_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c1241, k3, ← c1242, k3, ← c1243] at e_30_0
  have e_30_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c1244, ← c1245, k3, ← c1246] at e_30_1
  have e_30_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c1253, ← c1254, k3, ← c1255] at e_30_2
  refine ⟨?_, ?_⟩
  · have hc := e_29_12
    simp only [e_30_1] at hc
    linear_combination c1256.trans k1 - hc
  · have hc := e_30_2
    simp only [e_30_0, e_29_13, e_30_1] at hc
    linear_combination c1257.trans k1 - hc

theorem privateBatchWrapper4_f156 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 59) = band (a (.virt 165)) (a (.virt 167)) := by
  have c1258 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c1258, ← c1259, ← c1260] at e_29_14
  have hr := e_29_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f157 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 3) = band (a (.virt 169)) (a (.virt 171)) := by
  have c1261 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c1261, ← c1262, ← c1263] at e_31_0
  have hr := e_31_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f158 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 7) = band (a (.wire 29 59)) (a (.wire 31 3)) := by
  have c1264 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c1264, ← c1265, ← c1266] at e_31_1
  have hr := e_31_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f159 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 11) = band (a (.wire 29 23)) (a (.wire 31 7)) := by
  have c1267 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c1267, ← c1268, ← c1269] at e_31_2
  have hr := e_31_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f160 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 11) = a (.virt 11) := by
  have c1270 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1270

theorem privateBatchWrapper4_f161 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 35) = bnot (a (.wire 3 51)) := by
  have c278 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_6_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c278, k3, ← c279, k3, ← c280] at e_6_8
  have hr := e_6_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f162 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 23) = bnot (a (.wire 5 35)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c323, k3, ← c324, k3, ← c325] at e_8_5
  have hr := e_8_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f163 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 15) = band (a (.wire 6 35)) (a (.wire 8 23)) := by
  have c1271 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c1271, ← c1272, ← c1273] at e_31_3
  have hr := e_31_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f164 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 26)) (a (.virt 173)) (a (.virt 174)) := by
  have c1274 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (privateBatchWrapper4_copies40 a h).1
  have c1281 := (privateBatchWrapper4_copies40 a h).2.1
  have c1282 := (privateBatchWrapper4_copies40 a h).2.2.1
  have c1283 := (privateBatchWrapper4_copies40 a h).2.2.2.1
  have c1284 := (privateBatchWrapper4_copies40 a h).2.2.2.2.1
  have c1285 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.1
  have c1286 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.1
  have c1287 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_30_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c1274, k3, ← c1275, k3, ← c1276] at e_30_3
  have e_30_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c1277, ← c1278, k3, ← c1279] at e_30_4
  have e_30_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c1286, ← c1287, k3, ← c1288] at e_30_5
  have e_31_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c1280, ← c1281, ← c1282] at e_31_4
  have e_31_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c1283, ← c1284, ← c1285] at e_31_5
  refine ⟨?_, ?_⟩
  · have hc := e_31_4
    simp only [e_30_4] at hc
    linear_combination c1289.trans k1 - hc
  · have hc := e_30_5
    simp only [e_30_3, e_31_5, e_30_4] at hc
    linear_combination c1290.trans k1 - hc

theorem privateBatchWrapper4_f165 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 27)) (a (.virt 175)) (a (.virt 176)) := by
  have c1291 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_30_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c1291, k3, ← c1292, k3, ← c1293] at e_30_6
  have e_30_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c1294, ← c1295, k3, ← c1296] at e_30_7
  have e_30_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c1303, ← c1304, k3, ← c1305] at e_30_8
  have e_31_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c1297, ← c1298, ← c1299] at e_31_6
  have e_31_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c1300, ← c1301, ← c1302] at e_31_7
  refine ⟨?_, ?_⟩
  · have hc := e_31_6
    simp only [e_30_7] at hc
    linear_combination c1306.trans k1 - hc
  · have hc := e_30_8
    simp only [e_30_6, e_31_7, e_30_7] at hc
    linear_combination c1307.trans k1 - hc

theorem privateBatchWrapper4_f166 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 28)) (a (.virt 177)) (a (.virt 178)) := by
  have c1308 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1312 := (privateBatchWrapper4_copies41 a h).1
  have c1313 := (privateBatchWrapper4_copies41 a h).2.1
  have c1314 := (privateBatchWrapper4_copies41 a h).2.2.1
  have c1315 := (privateBatchWrapper4_copies41 a h).2.2.2.1
  have c1316 := (privateBatchWrapper4_copies41 a h).2.2.2.2.1
  have c1317 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.1
  have c1318 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.1
  have c1319 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.1
  have c1321 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have c1322 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_30_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c1308, k3, ← c1309, k3, ← c1310] at e_30_9
  have e_30_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c1311, ← c1312, k3, ← c1313] at e_30_10
  have e_30_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c1320, ← c1321, k3, ← c1322] at e_30_11
  have e_31_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c1314, ← c1315, ← c1316] at e_31_8
  have e_31_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c1317, ← c1318, ← c1319] at e_31_9
  refine ⟨?_, ?_⟩
  · have hc := e_31_8
    simp only [e_30_10] at hc
    linear_combination c1323.trans k1 - hc
  · have hc := e_30_11
    simp only [e_30_9, e_31_9, e_30_10] at hc
    linear_combination c1324.trans k1 - hc

theorem privateBatchWrapper4_f167 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 18)) (a (.virt 29)) (a (.virt 179)) (a (.virt 180)) := by
  have c1325 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1329 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1335 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_30_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c1325, k3, ← c1326, k3, ← c1327] at e_30_12
  have e_30_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c1328, ← c1329, k3, ← c1330] at e_30_13
  have e_30_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c1337, ← c1338, k3, ← c1339] at e_30_14
  have e_31_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c1331, ← c1332, ← c1333] at e_31_10
  have e_31_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c1334, ← c1335, ← c1336] at e_31_11
  refine ⟨?_, ?_⟩
  · have hc := e_31_10
    simp only [e_30_13] at hc
    linear_combination c1340.trans k1 - hc
  · have hc := e_30_14
    simp only [e_30_12, e_31_11, e_30_13] at hc
    linear_combination c1341.trans k1 - hc

theorem privateBatchWrapper4_f168 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 51) = band (a (.virt 173)) (a (.virt 175)) := by
  have c1342 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (privateBatchWrapper4_copies42 a h).1
  have e_31_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c1342, ← c1343, ← c1344] at e_31_12
  have hr := e_31_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f169 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 55) = band (a (.virt 177)) (a (.virt 179)) := by
  have c1345 := (privateBatchWrapper4_copies42 a h).2.1
  have c1346 := (privateBatchWrapper4_copies42 a h).2.2.1
  have c1347 := (privateBatchWrapper4_copies42 a h).2.2.2.1
  have e_31_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c1345, ← c1346, ← c1347] at e_31_13
  have hr := e_31_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f170 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 59) = band (a (.wire 31 51)) (a (.wire 31 55)) := by
  have c1348 := (privateBatchWrapper4_copies42 a h).2.2.2.2.1
  have c1349 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.1
  have c1350 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.1
  have e_31_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c1348, ← c1349, ← c1350] at e_31_14
  have hr := e_31_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f171 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 3) = band (a (.wire 31 15)) (a (.wire 31 59)) := by
  have c1351 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.1
  have c1352 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have e_32_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_0
  simp only [← c1351, ← c1352, ← c1353] at e_32_0
  have hr := e_32_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f172 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 3) = a (.virt 11) := by
  have c1354 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c1354

theorem privateBatchWrapper4_f173 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 7 19)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k3, ← c369, k3, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f174 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 7) = band (a (.wire 6 35)) (a (.wire 10 11)) := by
  have c1355 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_1
  simp only [← c1355, ← c1356, ← c1357] at e_32_1
  have hr := e_32_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f175 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 15)) (a (.virt 37)) (a (.virt 181)) (a (.virt 182)) := by
  have c1358 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1361 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1362 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1363 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_2
  simp only [← c1364, ← c1365, ← c1366] at e_32_2
  have e_32_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_3
  simp only [← c1367, ← c1368, ← c1369] at e_32_3
  have e_33_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c1358, k3, ← c1359, k3, ← c1360] at e_33_0
  have e_33_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c1361, ← c1362, k3, ← c1363] at e_33_1
  have e_33_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c1370, ← c1371, k3, ← c1372] at e_33_2
  refine ⟨?_, ?_⟩
  · have hc := e_32_2
    simp only [e_33_1] at hc
    linear_combination c1373.trans k1 - hc
  · have hc := e_33_2
    simp only [e_33_0, e_32_3, e_33_1] at hc
    linear_combination c1374.trans k1 - hc

theorem privateBatchWrapper4_f176 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 16)) (a (.virt 38)) (a (.virt 183)) (a (.virt 184)) := by
  have c1375 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (privateBatchWrapper4_copies43 a h).1
  have c1377 := (privateBatchWrapper4_copies43 a h).2.1
  have c1378 := (privateBatchWrapper4_copies43 a h).2.2.1
  have c1379 := (privateBatchWrapper4_copies43 a h).2.2.2.1
  have c1380 := (privateBatchWrapper4_copies43 a h).2.2.2.2.1
  have c1381 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.1
  have c1382 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.1
  have c1383 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have c1386 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1389 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_4
  simp only [← c1381, ← c1382, ← c1383] at e_32_4
  have e_32_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_5
  simp only [← c1384, ← c1385, ← c1386] at e_32_5
  have e_33_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c1375, k3, ← c1376, k3, ← c1377] at e_33_3
  have e_33_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c1378, ← c1379, k3, ← c1380] at e_33_4
  have e_33_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c1387, ← c1388, k3, ← c1389] at e_33_5
  refine ⟨?_, ?_⟩
  · have hc := e_32_4
    simp only [e_33_4] at hc
    linear_combination c1390.trans k1 - hc
  · have hc := e_33_5
    simp only [e_33_3, e_32_5, e_33_4] at hc
    linear_combination c1391.trans k1 - hc

theorem privateBatchWrapper4_f177 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17)) (a (.virt 39)) (a (.virt 185)) (a (.virt 186)) := by
  have c1392 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (privateBatchWrapper4_copies44 a h).1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_6
  simp only [← c1398, ← c1399, ← c1400] at e_32_6
  have e_32_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_7
  simp only [← c1401, ← c1402, ← c1403] at e_32_7
  have e_33_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c1392, k3, ← c1393, k3, ← c1394] at e_33_6
  have e_33_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1395, ← c1396, k3, ← c1397] at e_33_7
  have e_33_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c1404, ← c1405, k3, ← c1406] at e_33_8
  refine ⟨?_, ?_⟩
  · have hc := e_32_6
    simp only [e_33_7] at hc
    linear_combination c1407.trans k1 - hc
  · have hc := e_33_8
    simp only [e_33_6, e_32_7, e_33_7] at hc
    linear_combination c1408.trans k1 - hc

theorem privateBatchWrapper4_f178 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 18)) (a (.virt 40)) (a (.virt 187)) (a (.virt 188)) := by
  have c1409 := (privateBatchWrapper4_copies44 a h).2.1
  have c1410 := (privateBatchWrapper4_copies44 a h).2.2.1
  have c1411 := (privateBatchWrapper4_copies44 a h).2.2.2.1
  have c1412 := (privateBatchWrapper4_copies44 a h).2.2.2.2.1
  have c1413 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.1
  have c1414 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.1
  have c1417 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1423 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1425 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_32_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_8
  simp only [← c1415, ← c1416, ← c1417] at e_32_8
  have e_32_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_9
  simp only [← c1418, ← c1419, ← c1420] at e_32_9
  have e_33_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c1409, k3, ← c1410, k3, ← c1411] at e_33_9
  have e_33_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c1412, ← c1413, k3, ← c1414] at e_33_10
  have e_33_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c1421, ← c1422, k3, ← c1423] at e_33_11
  refine ⟨?_, ?_⟩
  · have hc := e_32_8
    simp only [e_33_10] at hc
    linear_combination c1424.trans k1 - hc
  · have hc := e_33_11
    simp only [e_33_9, e_32_9, e_33_10] at hc
    linear_combination c1425.trans k1 - hc

theorem privateBatchWrapper4_f179 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 43) = band (a (.virt 181)) (a (.virt 183)) := by
  have c1426 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1428 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_10
  simp only [← c1426, ← c1427, ← c1428] at e_32_10
  have hr := e_32_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f180 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 47) = band (a (.virt 185)) (a (.virt 187)) := by
  have c1429 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1431 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_11
  simp only [← c1429, ← c1430, ← c1431] at e_32_11
  have hr := e_32_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f181 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 51) = band (a (.wire 32 43)) (a (.wire 32 47)) := by
  have c1432 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1434 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_12
  simp only [← c1432, ← c1433, ← c1434] at e_32_12
  have hr := e_32_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f182 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 55) = band (a (.wire 32 7)) (a (.wire 32 51)) := by
  have c1435 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1437 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_13
  simp only [← c1435, ← c1436, ← c1437] at e_32_13
  have hr := e_32_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f183 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 55) = a (.virt 11) := by
  have c1438 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1438

theorem privateBatchWrapper4_f184 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 23) = bnot (a (.wire 5 35)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c323, k3, ← c324, k3, ← c325] at e_8_5
  have hr := e_8_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f185 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 7 19)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k3, ← c369, k3, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f186 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 59) = band (a (.wire 8 23)) (a (.wire 10 11)) := by
  have c1439 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1440 := (privateBatchWrapper4_copies45 a h).1
  have c1441 := (privateBatchWrapper4_copies45 a h).2.1
  have e_32_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_14
  simp only [← c1439, ← c1440, ← c1441] at e_32_14
  have hr := e_32_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f187 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 26)) (a (.virt 37)) (a (.virt 189)) (a (.virt 190)) := by
  have c1442 := (privateBatchWrapper4_copies45 a h).2.2.1
  have c1443 := (privateBatchWrapper4_copies45 a h).2.2.2.1
  have c1444 := (privateBatchWrapper4_copies45 a h).2.2.2.2.1
  have c1445 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.1
  have c1446 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.1
  have c1447 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.1
  have c1448 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1451 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1454 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1456 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_33_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c1442, k3, ← c1443, k3, ← c1444] at e_33_12
  have e_33_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c1445, ← c1446, k3, ← c1447] at e_33_13
  have e_33_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c1454, ← c1455, k3, ← c1456] at e_33_14
  have e_34_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c1448, ← c1449, ← c1450] at e_34_0
  have e_34_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_1
  simp only [← c1451, ← c1452, ← c1453] at e_34_1
  refine ⟨?_, ?_⟩
  · have hc := e_34_0
    simp only [e_33_13] at hc
    linear_combination c1457.trans k1 - hc
  · have hc := e_33_14
    simp only [e_33_12, e_34_1, e_33_13] at hc
    linear_combination c1458.trans k1 - hc

theorem privateBatchWrapper4_f188 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 27)) (a (.virt 38)) (a (.virt 191)) (a (.virt 192)) := by
  have c1459 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1465 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1468 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1469 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (privateBatchWrapper4_copies46 a h).1
  have c1473 := (privateBatchWrapper4_copies46 a h).2.1
  have c1474 := (privateBatchWrapper4_copies46 a h).2.2.1
  have c1475 := (privateBatchWrapper4_copies46 a h).2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_2
  simp only [← c1465, ← c1466, ← c1467] at e_34_2
  have e_34_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c1468, ← c1469, ← c1470] at e_34_3
  have e_35_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_0
  simp only [← c1459, k3, ← c1460, k3, ← c1461] at e_35_0
  have e_35_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_1
  simp only [← c1462, ← c1463, k3, ← c1464] at e_35_1
  have e_35_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_2
  simp only [← c1471, ← c1472, k3, ← c1473] at e_35_2
  refine ⟨?_, ?_⟩
  · have hc := e_34_2
    simp only [e_35_1] at hc
    linear_combination c1474.trans k1 - hc
  · have hc := e_35_2
    simp only [e_35_0, e_34_3, e_35_1] at hc
    linear_combination c1475.trans k1 - hc

theorem privateBatchWrapper4_f189 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 28)) (a (.virt 39)) (a (.virt 193)) (a (.virt 194)) := by
  have c1476 := (privateBatchWrapper4_copies46 a h).2.2.2.2.1
  have c1477 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.1
  have c1478 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.1
  have c1479 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.1
  have c1482 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1486 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_4
  simp only [← c1482, ← c1483, ← c1484] at e_34_4
  have e_34_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_5
  simp only [← c1485, ← c1486, ← c1487] at e_34_5
  have e_35_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_3
  simp only [← c1476, k3, ← c1477, k3, ← c1478] at e_35_3
  have e_35_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_4
  simp only [← c1479, ← c1480, k3, ← c1481] at e_35_4
  have e_35_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_5
  simp only [← c1488, ← c1489, k3, ← c1490] at e_35_5
  refine ⟨?_, ?_⟩
  · have hc := e_34_4
    simp only [e_35_4] at hc
    linear_combination c1491.trans k1 - hc
  · have hc := e_35_5
    simp only [e_35_3, e_34_5, e_35_4] at hc
    linear_combination c1492.trans k1 - hc

theorem privateBatchWrapper4_f190 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 29)) (a (.virt 40)) (a (.virt 195)) (a (.virt 196)) := by
  have c1493 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1496 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1499 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1502 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1503 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (privateBatchWrapper4_copies47 a h).1
  have c1505 := (privateBatchWrapper4_copies47 a h).2.1
  have c1506 := (privateBatchWrapper4_copies47 a h).2.2.1
  have c1507 := (privateBatchWrapper4_copies47 a h).2.2.2.1
  have c1508 := (privateBatchWrapper4_copies47 a h).2.2.2.2.1
  have c1509 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_34_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_6
  simp only [← c1499, ← c1500, ← c1501] at e_34_6
  have e_34_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_7
  simp only [← c1502, ← c1503, ← c1504] at e_34_7
  have e_35_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_6
  simp only [← c1493, k3, ← c1494, k3, ← c1495] at e_35_6
  have e_35_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_7
  simp only [← c1496, ← c1497, k3, ← c1498] at e_35_7
  have e_35_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_8
  simp only [← c1505, ← c1506, k3, ← c1507] at e_35_8
  refine ⟨?_, ?_⟩
  · have hc := e_34_6
    simp only [e_35_7] at hc
    linear_combination c1508.trans k1 - hc
  · have hc := e_35_8
    simp only [e_35_6, e_34_7, e_35_7] at hc
    linear_combination c1509.trans k1 - hc

theorem privateBatchWrapper4_f191 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 35) = band (a (.virt 189)) (a (.virt 191)) := by
  have c1510 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.1
  have e_34_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_8
  simp only [← c1510, ← c1511, ← c1512] at e_34_8
  have hr := e_34_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f192 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 39) = band (a (.virt 193)) (a (.virt 195)) := by
  have c1513 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have c1514 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_9
  simp only [← c1513, ← c1514, ← c1515] at e_34_9
  have hr := e_34_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f193 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 43) = band (a (.wire 34 35)) (a (.wire 34 39)) := by
  have c1516 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_10
  simp only [← c1516, ← c1517, ← c1518] at e_34_10
  have hr := e_34_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f194 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 47) = band (a (.wire 32 59)) (a (.wire 34 43)) := by
  have c1519 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1520 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_11
  simp only [← c1519, ← c1520, ← c1521] at e_34_11
  have hr := e_34_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f195 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 47) = a (.virt 11) := by
  have c1522 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1522

theorem privateBatchWrapper4_f196 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 7 19)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k3, ← c369, k3, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f197 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 36 12) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 0 ∧ a (.wire 36 13) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 1 ∧ a (.wire 36 14) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 2 ∧ a (.wire 36 15) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 3) := by
  have c1523 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1526 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1530 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1531 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1532 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1533 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 36 = addBlock (fun _ => 0) [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71), 1, 0, 0, 0] :=
    poseidon2In_first c1523.symm c1524.symm c1525.symm c1526.symm (c1527.symm.trans k3) (c1528.symm.trans k1) (c1529.symm.trans k1) (c1530.symm.trans k1) (c1531.symm.trans k1) (c1532.symm.trans k1) (c1533.symm.trans k1) (c1534.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 36) rfl rfl hin0
  have hpad : pad10 [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] =
      [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f198 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 37 12) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 0 ∧ a (.wire 37 13) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 1 ∧ a (.wire 37 14) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 2 ∧ a (.wire 37 15) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 3) := by
  have c1535 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (privateBatchWrapper4_copies48 a h).1
  have c1537 := (privateBatchWrapper4_copies48 a h).2.1
  have c1538 := (privateBatchWrapper4_copies48 a h).2.2.1
  have c1539 := (privateBatchWrapper4_copies48 a h).2.2.2.1
  have c1540 := (privateBatchWrapper4_copies48 a h).2.2.2.2.1
  have c1541 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.1
  have c1542 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.1
  have c1543 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.1
  have c1544 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.1
  have c1545 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have c1546 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 37 = addBlock (fun _ => 0) [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15), 1, 0, 0, 0] :=
    poseidon2In_first c1535.symm c1536.symm c1537.symm c1538.symm (c1539.symm.trans k3) (c1540.symm.trans k1) (c1541.symm.trans k1) (c1542.symm.trans k1) (c1543.symm.trans k1) (c1544.symm.trans k1) (c1545.symm.trans k1) (c1546.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 37) rfl rfl hin0
  have hpad : pad10 [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] =
      [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f199 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 35 43) = bselect (a (.wire 3 7)) (a (.wire 37 12)) (a (.virt 1)) := by
  have c1547 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1548 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_9
  simp only [← c1547, ← c1548, ← c1549] at e_35_9
  have e_35_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_10
  simp only [← c1550, ← c1551, ← c1552] at e_35_10
  have hr := e_35_10
  simp only [e_35_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f200 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 35 51) = bselect (a (.wire 3 7)) (a (.wire 37 13)) (a (.virt 2)) := by
  have c1553 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1554 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1556 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1557 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1558 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_11
  simp only [← c1553, ← c1554, ← c1555] at e_35_11
  have e_35_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_12
  simp only [← c1556, ← c1557, ← c1558] at e_35_12
  have hr := e_35_12
  simp only [e_35_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f201 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 35 59) = bselect (a (.wire 3 7)) (a (.wire 37 14)) (a (.virt 3)) := by
  have c1559 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1561 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1562 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_13
  simp only [← c1559, ← c1560, ← c1561] at e_35_13
  have e_35_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_14
  simp only [← c1562, ← c1563, ← c1564] at e_35_14
  have hr := e_35_14
  simp only [e_35_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f202 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 7) = bselect (a (.wire 3 7)) (a (.wire 37 15)) (a (.virt 4)) := by
  have c1565 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (privateBatchWrapper4_copies49 a h).1
  have c1569 := (privateBatchWrapper4_copies49 a h).2.1
  have c1570 := (privateBatchWrapper4_copies49 a h).2.2.1
  have e_38_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_0
  simp only [← c1565, ← c1566, ← c1567] at e_38_0
  have e_38_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_1
  simp only [← c1568, ← c1569, ← c1570] at e_38_1
  have hr := e_38_1
  simp only [e_38_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f203 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 39 12) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 0 ∧ a (.wire 39 13) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 1 ∧ a (.wire 39 14) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 2 ∧ a (.wire 39 15) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 3) := by
  have c1571 := (privateBatchWrapper4_copies49 a h).2.2.2.1
  have c1572 := (privateBatchWrapper4_copies49 a h).2.2.2.2.1
  have c1573 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.1
  have c1574 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.1
  have c1575 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have c1578 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1580 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1581 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 39 = addBlock (fun _ => 0) [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75), 1, 0, 0, 0] :=
    poseidon2In_first c1571.symm c1572.symm c1573.symm c1574.symm (c1575.symm.trans k3) (c1576.symm.trans k1) (c1577.symm.trans k1) (c1578.symm.trans k1) (c1579.symm.trans k1) (c1580.symm.trans k1) (c1581.symm.trans k1) (c1582.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 39) rfl rfl hin0
  have hpad : pad10 [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] =
      [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f204 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 40 12) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 0 ∧ a (.wire 40 13) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 1 ∧ a (.wire 40 14) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 2 ∧ a (.wire 40 15) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 3) := by
  have c1583 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1584 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1586 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1590 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1592 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1593 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 40 = addBlock (fun _ => 0) [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15), 1, 0, 0, 0] :=
    poseidon2In_first c1583.symm c1584.symm c1585.symm c1586.symm (c1587.symm.trans k3) (c1588.symm.trans k1) (c1589.symm.trans k1) (c1590.symm.trans k1) (c1591.symm.trans k1) (c1592.symm.trans k1) (c1593.symm.trans k1) (c1594.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 40) rfl rfl hin0
  have hpad : pad10 [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] =
      [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f205 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 15) = bselect (a (.wire 3 51)) (a (.wire 40 12)) (a (.virt 15)) := by
  have c1595 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1597 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1598 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1599 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (privateBatchWrapper4_copies50 a h).1
  have e_38_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_2
  simp only [← c1595, ← c1596, ← c1597] at e_38_2
  have e_38_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_3
  simp only [← c1598, ← c1599, ← c1600] at e_38_3
  have hr := e_38_3
  simp only [e_38_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f206 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 23) = bselect (a (.wire 3 51)) (a (.wire 40 13)) (a (.virt 16)) := by
  have c1601 := (privateBatchWrapper4_copies50 a h).2.1
  have c1602 := (privateBatchWrapper4_copies50 a h).2.2.1
  have c1603 := (privateBatchWrapper4_copies50 a h).2.2.2.1
  have c1604 := (privateBatchWrapper4_copies50 a h).2.2.2.2.1
  have c1605 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.1
  have c1606 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.1
  have e_38_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_4
  simp only [← c1601, ← c1602, ← c1603] at e_38_4
  have e_38_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_5
  simp only [← c1604, ← c1605, ← c1606] at e_38_5
  have hr := e_38_5
  simp only [e_38_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f207 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 31) = bselect (a (.wire 3 51)) (a (.wire 40 14)) (a (.virt 17)) := by
  have c1607 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.1
  have c1609 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have c1610 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1612 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_6
  simp only [← c1607, ← c1608, ← c1609] at e_38_6
  have e_38_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_7
  simp only [← c1610, ← c1611, ← c1612] at e_38_7
  have hr := e_38_7
  simp only [e_38_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f208 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 39) = bselect (a (.wire 3 51)) (a (.wire 40 15)) (a (.virt 18)) := by
  have c1613 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1614 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1616 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1617 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_8
  simp only [← c1613, ← c1614, ← c1615] at e_38_8
  have e_38_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_9
  simp only [← c1616, ← c1617, ← c1618] at e_38_9
  have hr := e_38_9
  simp only [e_38_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f209 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 41 12) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 0 ∧ a (.wire 41 13) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 1 ∧ a (.wire 41 14) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 2 ∧ a (.wire 41 15) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 3) := by
  have c1619 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1620 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1622 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1623 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1626 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1628 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1629 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1630 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 41 = addBlock (fun _ => 0) [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79), 1, 0, 0, 0] :=
    poseidon2In_first c1619.symm c1620.symm c1621.symm c1622.symm (c1623.symm.trans k3) (c1624.symm.trans k1) (c1625.symm.trans k1) (c1626.symm.trans k1) (c1627.symm.trans k1) (c1628.symm.trans k1) (c1629.symm.trans k1) (c1630.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 41) rfl rfl hin0
  have hpad : pad10 [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] =
      [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f210 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 42 12) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 0 ∧ a (.wire 42 13) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 1 ∧ a (.wire 42 14) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 2 ∧ a (.wire 42 15) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 3) := by
  have c1631 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (privateBatchWrapper4_copies51 a h).1
  have c1633 := (privateBatchWrapper4_copies51 a h).2.1
  have c1634 := (privateBatchWrapper4_copies51 a h).2.2.1
  have c1635 := (privateBatchWrapper4_copies51 a h).2.2.2.1
  have c1636 := (privateBatchWrapper4_copies51 a h).2.2.2.2.1
  have c1637 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.1
  have c1638 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.1
  have c1640 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 42 = addBlock (fun _ => 0) [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15), 1, 0, 0, 0] :=
    poseidon2In_first c1631.symm c1632.symm c1633.symm c1634.symm (c1635.symm.trans k3) (c1636.symm.trans k1) (c1637.symm.trans k1) (c1638.symm.trans k1) (c1639.symm.trans k1) (c1640.symm.trans k1) (c1641.symm.trans k1) (c1642.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 42) rfl rfl hin0
  have hpad : pad10 [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] =
      [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f211 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 47) = bselect (a (.wire 5 35)) (a (.wire 42 12)) (a (.virt 26)) := by
  have c1643 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1646 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_10
  simp only [← c1643, ← c1644, ← c1645] at e_38_10
  have e_38_11 := arithEq_of_rows h (i := 11) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_11
  simp only [← c1646, ← c1647, ← c1648] at e_38_11
  have hr := e_38_11
  simp only [e_38_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f212 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 38 55) = bselect (a (.wire 5 35)) (a (.wire 42 13)) (a (.virt 27)) := by
  have c1649 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_12 := arithEq_of_rows h (i := 12) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_12
  simp only [← c1649, ← c1650, ← c1651] at e_38_12
  have e_38_13 := arithEq_of_rows h (i := 13) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_13
  simp only [← c1652, ← c1653, ← c1654] at e_38_13
  have hr := e_38_13
  simp only [e_38_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f213 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 43 3) = bselect (a (.wire 5 35)) (a (.wire 42 14)) (a (.virt 28)) := by
  have c1655 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1657 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1658 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1660 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_14 := arithEq_of_rows h (i := 14) privateBatchWrapper4_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_14
  simp only [← c1655, ← c1656, ← c1657] at e_38_14
  have e_43_0 := arithEq_of_rows h (i := 0) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_0
  simp only [← c1658, ← c1659, ← c1660] at e_43_0
  have hr := e_43_0
  simp only [e_38_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f214 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 43 11) = bselect (a (.wire 5 35)) (a (.wire 42 15)) (a (.virt 29)) := by
  have c1661 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1663 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (privateBatchWrapper4_copies52 a h).1
  have c1665 := (privateBatchWrapper4_copies52 a h).2.1
  have c1666 := (privateBatchWrapper4_copies52 a h).2.2.1
  have e_43_1 := arithEq_of_rows h (i := 1) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_1
  simp only [← c1661, ← c1662, ← c1663] at e_43_1
  have e_43_2 := arithEq_of_rows h (i := 2) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_2
  simp only [← c1664, ← c1665, ← c1666] at e_43_2
  have hr := e_43_2
  simp only [e_43_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f215 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 44 12) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 0 ∧ a (.wire 44 13) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 1 ∧ a (.wire 44 14) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 2 ∧ a (.wire 44 15) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 3) := by
  have c1667 := (privateBatchWrapper4_copies52 a h).2.2.2.1
  have c1668 := (privateBatchWrapper4_copies52 a h).2.2.2.2.1
  have c1669 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.1
  have c1670 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.1
  have c1671 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.1
  have c1674 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1677 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 44 = addBlock (fun _ => 0) [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83), 1, 0, 0, 0] :=
    poseidon2In_first c1667.symm c1668.symm c1669.symm c1670.symm (c1671.symm.trans k3) (c1672.symm.trans k1) (c1673.symm.trans k1) (c1674.symm.trans k1) (c1675.symm.trans k1) (c1676.symm.trans k1) (c1677.symm.trans k1) (c1678.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 44) rfl rfl hin0
  have hpad : pad10 [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] =
      [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f216 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 45 12) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 0 ∧ a (.wire 45 13) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 1 ∧ a (.wire 45 14) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 2 ∧ a (.wire 45 15) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 3) := by
  have c1679 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1685 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1688 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1689 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper4_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 45 = addBlock (fun _ => 0) [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15), 1, 0, 0, 0] :=
    poseidon2In_first c1679.symm c1680.symm c1681.symm c1682.symm (c1683.symm.trans k3) (c1684.symm.trans k1) (c1685.symm.trans k1) (c1686.symm.trans k1) (c1687.symm.trans k1) (c1688.symm.trans k1) (c1689.symm.trans k1) (c1690.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 45) rfl rfl hin0
  have hpad : pad10 [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] =
      [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f217 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 43 19) = bselect (a (.wire 7 19)) (a (.wire 45 12)) (a (.virt 37)) := by
  have c1691 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1692 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1694 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (privateBatchWrapper4_copies53 a h).1
  have e_43_3 := arithEq_of_rows h (i := 3) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_3
  simp only [← c1691, ← c1692, ← c1693] at e_43_3
  have e_43_4 := arithEq_of_rows h (i := 4) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_4
  simp only [← c1694, ← c1695, ← c1696] at e_43_4
  have hr := e_43_4
  simp only [e_43_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f218 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 43 27) = bselect (a (.wire 7 19)) (a (.wire 45 13)) (a (.virt 38)) := by
  have c1697 := (privateBatchWrapper4_copies53 a h).2.1
  have c1698 := (privateBatchWrapper4_copies53 a h).2.2.1
  have c1699 := (privateBatchWrapper4_copies53 a h).2.2.2.1
  have c1700 := (privateBatchWrapper4_copies53 a h).2.2.2.2.1
  have c1701 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.1
  have c1702 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.1
  have e_43_5 := arithEq_of_rows h (i := 5) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_5
  simp only [← c1697, ← c1698, ← c1699] at e_43_5
  have e_43_6 := arithEq_of_rows h (i := 6) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_6
  simp only [← c1700, ← c1701, ← c1702] at e_43_6
  have hr := e_43_6
  simp only [e_43_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f219 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 43 35) = bselect (a (.wire 7 19)) (a (.wire 45 14)) (a (.virt 39)) := by
  have c1703 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.1
  have c1705 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1708 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_7 := arithEq_of_rows h (i := 7) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_7
  simp only [← c1703, ← c1704, ← c1705] at e_43_7
  have e_43_8 := arithEq_of_rows h (i := 8) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_8
  simp only [← c1706, ← c1707, ← c1708] at e_43_8
  have hr := e_43_8
  simp only [e_43_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f220 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 43 43) = bselect (a (.wire 7 19)) (a (.wire 45 15)) (a (.virt 40)) := by
  have c1709 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1711 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1714 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_9 := arithEq_of_rows h (i := 9) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_9
  simp only [← c1709, ← c1710, ← c1711] at e_43_9
  have e_43_10 := arithEq_of_rows h (i := 10) privateBatchWrapper4_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_10
  simp only [← c1712, ← c1713, ← c1714] at e_43_10
  have hr := e_43_10
  simp only [e_43_9] at hr
  simp only [bselect]
  linear_combination hr

end Plonky2Spec.Generated
