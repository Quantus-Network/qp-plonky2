/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PrivateBatchWrapper2`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge
import Plonky2Spec.Generated.PrivateBatchWrapper2.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem privateBatchWrapper2_f0 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 5)) (a (.virt 11)) (a (.virt 44)) (a (.virt 45)) := by
  have c0 := (privateBatchWrapper2_copies0 a h).1
  have c1 := (privateBatchWrapper2_copies0 a h).2.1
  have c2 := (privateBatchWrapper2_copies0 a h).2.2.1
  have c3 := (privateBatchWrapper2_copies0 a h).2.2.2.1
  have c4 := (privateBatchWrapper2_copies0 a h).2.2.2.2.1
  have c5 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.1
  have c6 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.1
  have c7 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c3, ← c4, ← c5] at e_1_12
  have e_1_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c6, ← c7, ← c8] at e_1_13
  have e_2_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c0, k2, ← c1, k2, ← c2] at e_2_0
  have e_2_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c9, ← c10, k2, ← c11] at e_2_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_12
    linear_combination c12.trans k1 - hc
  · have hc := e_2_1
    simp only [e_2_0, e_1_13] at hc
    linear_combination c13.trans k1 - hc

theorem privateBatchWrapper2_f1 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6)) (a (.virt 11)) (a (.virt 46)) (a (.virt 47)) := by
  have c14 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_1_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row1 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c17, ← c18, ← c19] at e_1_14
  have e_2_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c14, k2, ← c15, k2, ← c16] at e_2_2
  have e_2_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c23, ← c24, k2, ← c25] at e_2_3
  have e_3_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c20, ← c21, ← c22] at e_3_0
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_14
    linear_combination c26.trans k1 - hc
  · have hc := e_2_3
    simp only [e_2_2, e_3_0] at hc
    linear_combination c27.trans k1 - hc

theorem privateBatchWrapper2_f2 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 7)) (a (.virt 11)) (a (.virt 48)) (a (.virt 49)) := by
  have c28 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (privateBatchWrapper2_copies1 a h).1
  have c33 := (privateBatchWrapper2_copies1 a h).2.1
  have c34 := (privateBatchWrapper2_copies1 a h).2.2.1
  have c35 := (privateBatchWrapper2_copies1 a h).2.2.2.1
  have c36 := (privateBatchWrapper2_copies1 a h).2.2.2.2.1
  have c37 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.1
  have c38 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.1
  have c39 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c28, k2, ← c29, k2, ← c30] at e_2_4
  have e_2_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c37, ← c38, k2, ← c39] at e_2_5
  have e_3_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c31, ← c32, ← c33] at e_3_1
  have e_3_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c34, ← c35, ← c36] at e_3_2
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_1
    linear_combination c40.trans k1 - hc
  · have hc := e_2_5
    simp only [e_2_4, e_3_2] at hc
    linear_combination c41.trans k1 - hc

theorem privateBatchWrapper2_f3 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 8)) (a (.virt 11)) (a (.virt 50)) (a (.virt 51)) := by
  have c42 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c42, k2, ← c43, k2, ← c44] at e_2_6
  have e_2_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c51, ← c52, k2, ← c53] at e_2_7
  have e_3_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c45, ← c46, ← c47] at e_3_3
  have e_3_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c48, ← c49, ← c50] at e_3_4
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_3
    linear_combination c54.trans k1 - hc
  · have hc := e_2_7
    simp only [e_2_6, e_3_4] at hc
    linear_combination c55.trans k1 - hc

theorem privateBatchWrapper2_f4 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 23) = band (a (.virt 44)) (a (.virt 46)) := by
  have c56 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c56, ← c57, ← c58] at e_3_5
  have hr := e_3_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f5 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 27) = band (a (.virt 48)) (a (.virt 50)) := by
  have c59 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c59, ← c60, ← c61] at e_3_6
  have hr := e_3_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f6 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = band (a (.wire 3 23)) (a (.wire 3 27)) := by
  have c62 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (privateBatchWrapper2_copies2 a h).1
  have e_3_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c62, ← c63, ← c64] at e_3_7
  have hr := e_3_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f7 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 19)) (a (.virt 11)) (a (.virt 52)) (a (.virt 53)) := by
  have c65 := (privateBatchWrapper2_copies2 a h).2.1
  have c66 := (privateBatchWrapper2_copies2 a h).2.2.1
  have c67 := (privateBatchWrapper2_copies2 a h).2.2.2.1
  have c68 := (privateBatchWrapper2_copies2 a h).2.2.2.2.1
  have c69 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.1
  have c70 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.1
  have c71 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c65, k2, ← c66, k2, ← c67] at e_2_8
  have e_2_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c74, ← c75, k2, ← c76] at e_2_9
  have e_3_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c68, ← c69, ← c70] at e_3_8
  have e_3_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c71, ← c72, ← c73] at e_3_9
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_8
    linear_combination c77.trans k1 - hc
  · have hc := e_2_9
    simp only [e_2_8, e_3_9] at hc
    linear_combination c78.trans k1 - hc

theorem privateBatchWrapper2_f8 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 20)) (a (.virt 11)) (a (.virt 54)) (a (.virt 55)) := by
  have c79 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c79, k2, ← c80, k2, ← c81] at e_2_10
  have e_2_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c88, ← c89, k2, ← c90] at e_2_11
  have e_3_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c82, ← c83, ← c84] at e_3_10
  have e_3_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c85, ← c86, ← c87] at e_3_11
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_10
    linear_combination c91.trans k1 - hc
  · have hc := e_2_11
    simp only [e_2_10, e_3_11] at hc
    linear_combination c92.trans k1 - hc

theorem privateBatchWrapper2_f9 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 21)) (a (.virt 11)) (a (.virt 56)) (a (.virt 57)) := by
  have c93 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (privateBatchWrapper2_copies3 a h).1
  have c97 := (privateBatchWrapper2_copies3 a h).2.1
  have c98 := (privateBatchWrapper2_copies3 a h).2.2.1
  have c99 := (privateBatchWrapper2_copies3 a h).2.2.2.1
  have c100 := (privateBatchWrapper2_copies3 a h).2.2.2.2.1
  have c101 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.1
  have c102 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.1
  have c103 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c93, k2, ← c94, k2, ← c95] at e_2_12
  have e_2_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c102, ← c103, k2, ← c104] at e_2_13
  have e_3_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c96, ← c97, ← c98] at e_3_12
  have e_3_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c99, ← c100, ← c101] at e_3_13
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_12
    linear_combination c105.trans k1 - hc
  · have hc := e_2_13
    simp only [e_2_12, e_3_13] at hc
    linear_combination c106.trans k1 - hc

theorem privateBatchWrapper2_f10 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 22)) (a (.virt 11)) (a (.virt 58)) (a (.virt 59)) := by
  have c107 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_2_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c107, k2, ← c108, k2, ← c109] at e_2_14
  have e_3_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c110, ← c111, ← c112] at e_3_14
  have e_4_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c113, ← c114, ← c115] at e_4_0
  have e_5_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c116, ← c117, k2, ← c118] at e_5_0
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_3_14
    linear_combination c119.trans k1 - hc
  · have hc := e_5_0
    simp only [e_2_14, e_4_0] at hc
    linear_combination c120.trans k1 - hc

theorem privateBatchWrapper2_f11 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 7) = band (a (.virt 52)) (a (.virt 54)) := by
  have c121 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c121, ← c122, ← c123] at e_4_1
  have hr := e_4_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f12 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 11) = band (a (.virt 56)) (a (.virt 58)) := by
  have c124 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c124, ← c125, ← c126] at e_4_2
  have hr := e_4_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f13 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11)) := by
  have c127 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (privateBatchWrapper2_copies4 a h).1
  have c129 := (privateBatchWrapper2_copies4 a h).2.1
  have e_4_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c127, ← c128, ← c129] at e_4_3
  have hr := e_4_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f14 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 7) = bnot (a (.wire 3 31)) := by
  have c130 := (privateBatchWrapper2_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper2_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper2_copies4 a h).2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c130, k2, ← c131, k2, ← c132] at e_5_1
  have hr := e_5_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f15 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 12) = bnot (a (.virt 11)) := by
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [bnot, k1, k2]
  ring

theorem privateBatchWrapper2_f16 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 7) = band (a (.wire 5 7)) (a (.virt 12)) := by
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [band, k2]
  ring

theorem privateBatchWrapper2_f17 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 11) = bselect (a (.wire 5 7)) (a (.virt 5)) (a (.virt 11)) := by
  have c133 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.1
  have c134 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.1
  have c135 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c133, ← c134, ← c135, k1] at e_5_2
  have hr := e_5_2
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f18 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 15) = bselect (a (.wire 5 7)) (a (.virt 6)) (a (.virt 11)) := by
  have c136 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c136, ← c137, ← c138, k1] at e_5_3
  have hr := e_5_3
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f19 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 19) = bselect (a (.wire 5 7)) (a (.virt 7)) (a (.virt 11)) := by
  have c139 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c139, ← c140, ← c141, k1] at e_5_4
  have hr := e_5_4
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f20 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 23) = bselect (a (.wire 5 7)) (a (.virt 8)) (a (.virt 11)) := by
  have c142 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c144 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c142, ← c143, ← c144, k1] at e_5_5
  have hr := e_5_5
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f21 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 27) = bselect (a (.wire 5 7)) (a (.virt 9)) (a (.virt 11)) := by
  have c145 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c145, ← c146, ← c147, k1] at e_5_6
  have hr := e_5_6
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f22 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 7) = bor (a (.virt 11)) (a (.wire 5 7)) := by
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [bor, k1]
  ring

theorem privateBatchWrapper2_f23 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 31) = bnot (a (.wire 4 15)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c148, k2, ← c149, k2, ← c150] at e_5_7
  have hr := e_5_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f24 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 35) = bnot (a (.wire 5 7)) := by
  have c151 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c151, k2, ← c152, k2, ← c153] at e_5_8
  have hr := e_5_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f25 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 19) = band (a (.wire 5 31)) (a (.wire 5 35)) := by
  have c154 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c154, ← c155, ← c156] at e_4_4
  have hr := e_4_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f26 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 43) = bselect (a (.wire 4 19)) (a (.virt 19)) (a (.wire 5 11)) := by
  have c157 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (privateBatchWrapper2_copies5 a h).1
  have c161 := (privateBatchWrapper2_copies5 a h).2.1
  have c162 := (privateBatchWrapper2_copies5 a h).2.2.1
  have e_5_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c157, ← c158, ← c159] at e_5_9
  have e_5_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c160, ← c161, ← c162] at e_5_10
  have hr := e_5_10
  simp only [e_5_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f27 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 51) = bselect (a (.wire 4 19)) (a (.virt 20)) (a (.wire 5 15)) := by
  have c163 := (privateBatchWrapper2_copies5 a h).2.2.2.1
  have c164 := (privateBatchWrapper2_copies5 a h).2.2.2.2.1
  have c165 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.1
  have c166 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.1
  have c167 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.1
  have e_5_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c163, ← c164, ← c165] at e_5_11
  have e_5_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c166, ← c167, ← c168] at e_5_12
  have hr := e_5_12
  simp only [e_5_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f28 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 59) = bselect (a (.wire 4 19)) (a (.virt 21)) (a (.wire 5 19)) := by
  have c169 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c169, ← c170, ← c171] at e_5_13
  have e_5_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c172, ← c173, ← c174] at e_5_14
  have hr := e_5_14
  simp only [e_5_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f29 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 7) = bselect (a (.wire 4 19)) (a (.virt 22)) (a (.wire 5 23)) := by
  have c175 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c175, ← c176, ← c177] at e_6_0
  have e_6_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c178, ← c179, ← c180] at e_6_1
  have hr := e_6_1
  simp only [e_6_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f30 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 15) = bselect (a (.wire 4 19)) (a (.virt 23)) (a (.wire 5 27)) := by
  have c181 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c181, ← c182, ← c183] at e_6_2
  have e_6_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c184, ← c185, ← c186] at e_6_3
  have hr := e_6_3
  simp only [e_6_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f31 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 0 59) = bor (a (.wire 5 7)) (a (.wire 5 31)) := by
  have c187 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (privateBatchWrapper2_copies6 a h).1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_0_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c190, ← c191, k2, ← c192] at e_0_14
  have e_7_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c187, ← c188, ← c189] at e_7_0
  have hr := e_0_14
  simp only [e_7_0] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f32 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 5)) (a (.wire 5 43)) (a (.virt 60)) (a (.virt 61)) := by
  have c193 := (privateBatchWrapper2_copies6 a h).2.1
  have c194 := (privateBatchWrapper2_copies6 a h).2.2.1
  have c195 := (privateBatchWrapper2_copies6 a h).2.2.2.1
  have c196 := (privateBatchWrapper2_copies6 a h).2.2.2.2.1
  have c197 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.1
  have c198 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.1
  have c199 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c199, ← c200, ← c201] at e_4_5
  have e_4_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c202, ← c203, ← c204] at e_4_6
  have e_6_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c193, k2, ← c194, k2, ← c195] at e_6_4
  have e_6_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c196, ← c197, k2, ← c198] at e_6_5
  have e_6_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c205, ← c206, k2, ← c207] at e_6_6
  refine ⟨?_, ?_⟩
  · have hc := e_4_5
    simp only [e_6_5] at hc
    linear_combination c208.trans k1 - hc
  · have hc := e_6_6
    simp only [e_6_4, e_4_6, e_6_5] at hc
    linear_combination c209.trans k1 - hc

theorem privateBatchWrapper2_f33 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6)) (a (.wire 5 51)) (a (.virt 62)) (a (.virt 63)) := by
  have c210 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (privateBatchWrapper2_copies7 a h).1
  have c225 := (privateBatchWrapper2_copies7 a h).2.1
  have c226 := (privateBatchWrapper2_copies7 a h).2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c216, ← c217, ← c218] at e_4_7
  have e_4_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c219, ← c220, ← c221] at e_4_8
  have e_6_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c210, k2, ← c211, k2, ← c212] at e_6_7
  have e_6_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c213, ← c214, k2, ← c215] at e_6_8
  have e_6_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c222, ← c223, k2, ← c224] at e_6_9
  refine ⟨?_, ?_⟩
  · have hc := e_4_7
    simp only [e_6_8] at hc
    linear_combination c225.trans k1 - hc
  · have hc := e_6_9
    simp only [e_6_7, e_4_8, e_6_8] at hc
    linear_combination c226.trans k1 - hc

theorem privateBatchWrapper2_f34 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 7)) (a (.wire 5 59)) (a (.virt 64)) (a (.virt 65)) := by
  have c227 := (privateBatchWrapper2_copies7 a h).2.2.2.1
  have c228 := (privateBatchWrapper2_copies7 a h).2.2.2.2.1
  have c229 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.1
  have c230 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.1
  have c231 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c233, ← c234, ← c235] at e_4_9
  have e_4_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c236, ← c237, ← c238] at e_4_10
  have e_6_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c227, k2, ← c228, k2, ← c229] at e_6_10
  have e_6_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c230, ← c231, k2, ← c232] at e_6_11
  have e_6_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c239, ← c240, k2, ← c241] at e_6_12
  refine ⟨?_, ?_⟩
  · have hc := e_4_9
    simp only [e_6_11] at hc
    linear_combination c242.trans k1 - hc
  · have hc := e_6_12
    simp only [e_6_10, e_4_10, e_6_11] at hc
    linear_combination c243.trans k1 - hc

theorem privateBatchWrapper2_f35 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 8)) (a (.wire 6 7)) (a (.virt 66)) (a (.virt 67)) := by
  have c244 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (privateBatchWrapper2_copies8 a h).1
  have c257 := (privateBatchWrapper2_copies8 a h).2.1
  have c258 := (privateBatchWrapper2_copies8 a h).2.2.1
  have c259 := (privateBatchWrapper2_copies8 a h).2.2.2.1
  have c260 := (privateBatchWrapper2_copies8 a h).2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_4_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c250, ← c251, ← c252] at e_4_11
  have e_4_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c253, ← c254, ← c255] at e_4_12
  have e_6_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c244, k2, ← c245, k2, ← c246] at e_6_13
  have e_6_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c247, ← c248, k2, ← c249] at e_6_14
  have e_8_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c256, ← c257, k2, ← c258] at e_8_0
  refine ⟨?_, ?_⟩
  · have hc := e_4_11
    simp only [e_6_14] at hc
    linear_combination c259.trans k1 - hc
  · have hc := e_8_0
    simp only [e_6_13, e_4_12, e_6_14] at hc
    linear_combination c260.trans k1 - hc

theorem privateBatchWrapper2_f36 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 55) = band (a (.virt 60)) (a (.virt 62)) := by
  have c261 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.1
  have c263 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.1
  have e_4_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c261, ← c262, ← c263] at e_4_13
  have hr := e_4_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f37 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 59) = band (a (.virt 64)) (a (.virt 66)) := by
  have c264 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_4_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c264, ← c265, ← c266] at e_4_14
  have hr := e_4_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f38 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 3) = band (a (.wire 4 55)) (a (.wire 4 59)) := by
  have c267 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c267, ← c268, ← c269] at e_9_0
  have hr := e_9_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f39 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 3) = bor (a (.wire 3 31)) (a (.wire 9 3)) := by
  have c270 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c270, ← c271, ← c272] at e_7_1
  have e_10_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c273, ← c274, k2, ← c275] at e_10_0
  have hr := e_10_0
  simp only [e_7_1] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f40 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 3) = a (.virt 12) := by
  have c276 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c276

theorem privateBatchWrapper2_f42 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 19)) (a (.wire 5 43)) (a (.virt 68)) (a (.virt 69)) := by
  have c278 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (privateBatchWrapper2_copies9 a h).1
  have c289 := (privateBatchWrapper2_copies9 a h).2.1
  have c290 := (privateBatchWrapper2_copies9 a h).2.2.1
  have c291 := (privateBatchWrapper2_copies9 a h).2.2.2.1
  have c292 := (privateBatchWrapper2_copies9 a h).2.2.2.2.1
  have c293 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.1
  have c294 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c278, k2, ← c279, k2, ← c280] at e_8_1
  have e_8_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c281, ← c282, k2, ← c283] at e_8_2
  have e_8_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c290, ← c291, k2, ← c292] at e_8_3
  have e_9_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c284, ← c285, ← c286] at e_9_1
  have e_9_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c287, ← c288, ← c289] at e_9_2
  refine ⟨?_, ?_⟩
  · have hc := e_9_1
    simp only [e_8_2] at hc
    linear_combination c293.trans k1 - hc
  · have hc := e_8_3
    simp only [e_8_1, e_9_2, e_8_2] at hc
    linear_combination c294.trans k1 - hc

theorem privateBatchWrapper2_f43 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 20)) (a (.wire 5 51)) (a (.virt 70)) (a (.virt 71)) := by
  have c295 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c295, k2, ← c296, k2, ← c297] at e_8_4
  have e_8_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c298, ← c299, k2, ← c300] at e_8_5
  have e_8_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c307, ← c308, k2, ← c309] at e_8_6
  have e_9_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c301, ← c302, ← c303] at e_9_3
  have e_9_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c304, ← c305, ← c306] at e_9_4
  refine ⟨?_, ?_⟩
  · have hc := e_9_3
    simp only [e_8_5] at hc
    linear_combination c310.trans k1 - hc
  · have hc := e_8_6
    simp only [e_8_4, e_9_4, e_8_5] at hc
    linear_combination c311.trans k1 - hc

theorem privateBatchWrapper2_f44 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 21)) (a (.wire 5 59)) (a (.virt 72)) (a (.virt 73)) := by
  have c312 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (privateBatchWrapper2_copies10 a h).1
  have c321 := (privateBatchWrapper2_copies10 a h).2.1
  have c322 := (privateBatchWrapper2_copies10 a h).2.2.1
  have c323 := (privateBatchWrapper2_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper2_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.1
  have c326 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.1
  have c327 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c312, k2, ← c313, k2, ← c314] at e_8_7
  have e_8_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c315, ← c316, k2, ← c317] at e_8_8
  have e_8_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c324, ← c325, k2, ← c326] at e_8_9
  have e_9_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c318, ← c319, ← c320] at e_9_5
  have e_9_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c321, ← c322, ← c323] at e_9_6
  refine ⟨?_, ?_⟩
  · have hc := e_9_5
    simp only [e_8_8] at hc
    linear_combination c327.trans k1 - hc
  · have hc := e_8_9
    simp only [e_8_7, e_9_6, e_8_8] at hc
    linear_combination c328.trans k1 - hc

theorem privateBatchWrapper2_f45 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 22)) (a (.wire 6 7)) (a (.virt 74)) (a (.virt 75)) := by
  have c329 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c329, k2, ← c330, k2, ← c331] at e_8_10
  have e_8_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c332, ← c333, k2, ← c334] at e_8_11
  have e_8_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c341, ← c342, k2, ← c343] at e_8_12
  have e_9_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c335, ← c336, ← c337] at e_9_7
  have e_9_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c338, ← c339, ← c340] at e_9_8
  refine ⟨?_, ?_⟩
  · have hc := e_9_7
    simp only [e_8_11] at hc
    linear_combination c344.trans k1 - hc
  · have hc := e_8_12
    simp only [e_8_10, e_9_8, e_8_11] at hc
    linear_combination c345.trans k1 - hc

theorem privateBatchWrapper2_f46 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 39) = band (a (.virt 68)) (a (.virt 70)) := by
  have c346 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c346, ← c347, ← c348] at e_9_9
  have hr := e_9_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f47 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 43) = band (a (.virt 72)) (a (.virt 74)) := by
  have c349 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_9_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c349, ← c350, ← c351] at e_9_10
  have hr := e_9_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f48 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 47) = band (a (.wire 9 39)) (a (.wire 9 43)) := by
  have c352 := (privateBatchWrapper2_copies11 a h).1
  have c353 := (privateBatchWrapper2_copies11 a h).2.1
  have c354 := (privateBatchWrapper2_copies11 a h).2.2.1
  have e_9_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c352, ← c353, ← c354] at e_9_11
  have hr := e_9_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f49 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 7) = bor (a (.wire 4 15)) (a (.wire 9 47)) := by
  have c355 := (privateBatchWrapper2_copies11 a h).2.2.2.1
  have c356 := (privateBatchWrapper2_copies11 a h).2.2.2.2.1
  have c357 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.1
  have c358 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.1
  have c359 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_7_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c355, ← c356, ← c357] at e_7_2
  have e_10_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c358, ← c359, k2, ← c360] at e_10_1
  have hr := e_10_1
  simp only [e_7_2] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f50 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 7) = a (.virt 12) := by
  have c361 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.1
  exact c361

theorem privateBatchWrapper2_f51 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 14) = a (.virt 0) := by
  have c362 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c362

theorem privateBatchWrapper2_f52 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 59) = bselect (a (.wire 3 31)) (a (.virt 11)) (a (.virt 10)) := by
  have c363 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_8_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c363, ← c364, ← c365] at e_8_13
  have e_8_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c366, ← c367, k1, ← c368] at e_8_14
  have hr := e_8_14
  simp only [e_8_13] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f53 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 59) = a (.virt 11) + a (.wire 8 59) := by
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [k1]
  ring

theorem privateBatchWrapper2_f54 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 7) = bselect (a (.wire 4 15)) (a (.virt 11)) (a (.virt 24)) := by
  have c369 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c369, ← c370, ← c371] at e_11_0
  have e_11_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c372, ← c373, k1, ← c374] at e_11_1
  have hr := e_11_1
  simp only [e_11_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f55 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 11) = a (.wire 8 59) + a (.wire 11 7) := by
  have c375 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c375, ← c376, k2, ← c377] at e_10_2
  have hr := e_10_2
  linear_combination hr

theorem privateBatchWrapper2_f56 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.virt 30)) 32 := by
  have c378 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (privateBatchWrapper2_copies12 a h).1
  have c385 := (privateBatchWrapper2_copies12 a h).2.1
  have c386 := (privateBatchWrapper2_copies12 a h).2.2.1
  have c387 := (privateBatchWrapper2_copies12 a h).2.2.2.1
  have c388 := (privateBatchWrapper2_copies12 a h).2.2.2.2.1
  have c389 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.1
  have c390 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.1
  have c391 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c405 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 12) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c378.trans k1
    · exact c379.trans k1
    · exact c380.trans k1
    · exact c381.trans k1
    · exact c382.trans k1
    · exact c383.trans k1
    · exact c384.trans k1
    · exact c385.trans k1
    · exact c386.trans k1
    · exact c387.trans k1
    · exact c388.trans k1
    · exact c389.trans k1
    · exact c390.trans k1
    · exact c391.trans k1
    · exact c392.trans k1
    · exact c393.trans k1
    · exact c394.trans k1
    · exact c395.trans k1
    · exact c396.trans k1
    · exact c397.trans k1
    · exact c398.trans k1
    · exact c399.trans k1
    · exact c400.trans k1
    · exact c401.trans k1
    · exact c402.trans k1
    · exact c403.trans k1
    · exact c404.trans k1
    )
  rwa [c405] at hr

theorem privateBatchWrapper2_f57 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 30) = a (.virt 11) + a (.virt 30) := by
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  simp only [k1]
  ring

theorem privateBatchWrapper2_f58 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.virt 35)) 32 := by
  have c406 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (privateBatchWrapper2_copies13 a h).1
  have c417 := (privateBatchWrapper2_copies13 a h).2.1
  have c418 := (privateBatchWrapper2_copies13 a h).2.2.1
  have c419 := (privateBatchWrapper2_copies13 a h).2.2.2.1
  have c420 := (privateBatchWrapper2_copies13 a h).2.2.2.2.1
  have c421 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.1
  have c422 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.1
  have c423 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 13) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c406.trans k1
    · exact c407.trans k1
    · exact c408.trans k1
    · exact c409.trans k1
    · exact c410.trans k1
    · exact c411.trans k1
    · exact c412.trans k1
    · exact c413.trans k1
    · exact c414.trans k1
    · exact c415.trans k1
    · exact c416.trans k1
    · exact c417.trans k1
    · exact c418.trans k1
    · exact c419.trans k1
    · exact c420.trans k1
    · exact c421.trans k1
    · exact c422.trans k1
    · exact c423.trans k1
    · exact c424.trans k1
    · exact c425.trans k1
    · exact c426.trans k1
    · exact c427.trans k1
    · exact c428.trans k1
    · exact c429.trans k1
    · exact c430.trans k1
    · exact c431.trans k1
    · exact c432.trans k1
    )
  rwa [c433] at hr

theorem privateBatchWrapper2_f59 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 15) = a (.virt 30) + a (.virt 35) := by
  have c434 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_10_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c434, ← c435, k2, ← c436] at e_10_3
  have hr := e_10_3
  linear_combination hr

theorem privateBatchWrapper2_f60 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.virt 25)) 14 := by
  have c437 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (privateBatchWrapper2_copies14 a h).1
  have c449 := (privateBatchWrapper2_copies14 a h).2.1
  have c450 := (privateBatchWrapper2_copies14 a h).2.2.1
  have c451 := (privateBatchWrapper2_copies14 a h).2.2.2.1
  have c452 := (privateBatchWrapper2_copies14 a h).2.2.2.2.1
  have c453 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.1
  have c454 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.1
  have c455 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (privateBatchWrapper2_copies15 a h).1
  have c481 := (privateBatchWrapper2_copies15 a h).2.1
  have c482 := (privateBatchWrapper2_copies15 a h).2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 14) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c437.trans k1
    · exact c438.trans k1
    · exact c439.trans k1
    · exact c440.trans k1
    · exact c441.trans k1
    · exact c442.trans k1
    · exact c443.trans k1
    · exact c444.trans k1
    · exact c445.trans k1
    · exact c446.trans k1
    · exact c447.trans k1
    · exact c448.trans k1
    · exact c449.trans k1
    · exact c450.trans k1
    · exact c451.trans k1
    · exact c452.trans k1
    · exact c453.trans k1
    · exact c454.trans k1
    · exact c455.trans k1
    · exact c456.trans k1
    · exact c457.trans k1
    · exact c458.trans k1
    · exact c459.trans k1
    · exact c460.trans k1
    · exact c461.trans k1
    · exact c462.trans k1
    · exact c463.trans k1
    · exact c464.trans k1
    · exact c465.trans k1
    · exact c466.trans k1
    · exact c467.trans k1
    · exact c468.trans k1
    · exact c469.trans k1
    · exact c470.trans k1
    · exact c471.trans k1
    · exact c472.trans k1
    · exact c473.trans k1
    · exact c474.trans k1
    · exact c475.trans k1
    · exact c476.trans k1
    · exact c477.trans k1
    · exact c478.trans k1
    · exact c479.trans k1
    · exact c480.trans k1
    · exact c481.trans k1
    )
  rwa [c482] at hr

theorem privateBatchWrapper2_f61 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 11) = a (.virt 76) - a (.virt 25) := by
  have c483 := (privateBatchWrapper2_copies15 a h).2.2.2.1
  have c484 := (privateBatchWrapper2_copies15 a h).2.2.2.2.1
  have c485 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c483, k0, ← c484, k2, ← c485] at e_11_2
  have hr := e_11_2
  simp only [k0]
  linear_combination hr

theorem privateBatchWrapper2_f62 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 11 11)) 14 := by
  have c486 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.1
  have c487 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (privateBatchWrapper2_copies16 a h).1
  have c513 := (privateBatchWrapper2_copies16 a h).2.1
  have c514 := (privateBatchWrapper2_copies16 a h).2.2.1
  have c515 := (privateBatchWrapper2_copies16 a h).2.2.2.1
  have c516 := (privateBatchWrapper2_copies16 a h).2.2.2.2.1
  have c517 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.1
  have c518 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.1
  have c519 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 15) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c486.trans k1
    · exact c487.trans k1
    · exact c488.trans k1
    · exact c489.trans k1
    · exact c490.trans k1
    · exact c491.trans k1
    · exact c492.trans k1
    · exact c493.trans k1
    · exact c494.trans k1
    · exact c495.trans k1
    · exact c496.trans k1
    · exact c497.trans k1
    · exact c498.trans k1
    · exact c499.trans k1
    · exact c500.trans k1
    · exact c501.trans k1
    · exact c502.trans k1
    · exact c503.trans k1
    · exact c504.trans k1
    · exact c505.trans k1
    · exact c506.trans k1
    · exact c507.trans k1
    · exact c508.trans k1
    · exact c509.trans k1
    · exact c510.trans k1
    · exact c511.trans k1
    · exact c512.trans k1
    · exact c513.trans k1
    · exact c514.trans k1
    · exact c515.trans k1
    · exact c516.trans k1
    · exact c517.trans k1
    · exact c518.trans k1
    · exact c519.trans k1
    · exact c520.trans k1
    · exact c521.trans k1
    · exact c522.trans k1
    · exact c523.trans k1
    · exact c524.trans k1
    · exact c525.trans k1
    · exact c526.trans k1
    · exact c527.trans k1
    · exact c528.trans k1
    · exact c529.trans k1
    · exact c530.trans k1
    )
  rwa [c531] at hr

theorem privateBatchWrapper2_f63 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 51) = a (.wire 10 15) * a (.virt 76) := by
  have c532 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_9_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c532, ← c533, k0, ← c534] at e_9_12
  have hr := e_9_12
  simp only [k0]
  linear_combination hr

theorem privateBatchWrapper2_f64 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 55) = a (.wire 10 11) * a (.wire 11 11) := by
  have c535 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c535, ← c536, ← c537] at e_9_13
  have hr := e_9_13
  linear_combination hr

theorem privateBatchWrapper2_f65 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 15) = a (.wire 9 55) - a (.wire 9 51) := by
  have c538 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c538, ← c539, k2, ← c540] at e_11_3
  have hr := e_11_3
  linear_combination hr

theorem privateBatchWrapper2_f66 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 11 15)) 52 := by
  have c541 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (privateBatchWrapper2_copies17 a h).1
  have c545 := (privateBatchWrapper2_copies17 a h).2.1
  have c546 := (privateBatchWrapper2_copies17 a h).2.2.1
  have c547 := (privateBatchWrapper2_copies17 a h).2.2.2.1
  have c548 := (privateBatchWrapper2_copies17 a h).2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hr := rangeCheck_of_row h (row := 16) (N := 59) (n := 52) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c541.trans k1
    · exact c542.trans k1
    · exact c543.trans k1
    · exact c544.trans k1
    · exact c545.trans k1
    · exact c546.trans k1
    · exact c547.trans k1
    )
  rwa [c548] at hr

theorem privateBatchWrapper2_f67 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 7) = bnot (a (.wire 3 31)) := by
  have c130 := (privateBatchWrapper2_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper2_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper2_copies4 a h).2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c130, k2, ← c131, k2, ← c132] at e_5_1
  have hr := e_5_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f68 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 31) = bnot (a (.wire 4 15)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c148, k2, ← c149, k2, ← c150] at e_5_7
  have hr := e_5_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f69 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 59) = band (a (.wire 5 7)) (a (.wire 5 31)) := by
  have c549 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.1
  have c550 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.1
  have c551 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.1
  have e_9_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c549, ← c550, ← c551] at e_9_14
  have hr := e_9_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f70 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 1)) (a (.virt 15)) (a (.virt 77)) (a (.virt 78)) := by
  have c552 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c552, k2, ← c553, k2, ← c554] at e_11_4
  have e_11_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c555, ← c556, k2, ← c557] at e_11_5
  have e_11_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c564, ← c565, k2, ← c566] at e_11_6
  have e_17_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c558, ← c559, ← c560] at e_17_0
  have e_17_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c561, ← c562, ← c563] at e_17_1
  refine ⟨?_, ?_⟩
  · have hc := e_17_0
    simp only [e_11_5] at hc
    linear_combination c567.trans k1 - hc
  · have hc := e_11_6
    simp only [e_11_4, e_17_1, e_11_5] at hc
    linear_combination c568.trans k1 - hc

theorem privateBatchWrapper2_f71 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 2)) (a (.virt 16)) (a (.virt 79)) (a (.virt 80)) := by
  have c569 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (privateBatchWrapper2_copies18 a h).1
  have c577 := (privateBatchWrapper2_copies18 a h).2.1
  have c578 := (privateBatchWrapper2_copies18 a h).2.2.1
  have c579 := (privateBatchWrapper2_copies18 a h).2.2.2.1
  have c580 := (privateBatchWrapper2_copies18 a h).2.2.2.2.1
  have c581 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.1
  have c582 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.1
  have c583 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c569, k2, ← c570, k2, ← c571] at e_11_7
  have e_11_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c572, ← c573, k2, ← c574] at e_11_8
  have e_11_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c581, ← c582, k2, ← c583] at e_11_9
  have e_17_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c575, ← c576, ← c577] at e_17_2
  have e_17_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c578, ← c579, ← c580] at e_17_3
  refine ⟨?_, ?_⟩
  · have hc := e_17_2
    simp only [e_11_8] at hc
    linear_combination c584.trans k1 - hc
  · have hc := e_11_9
    simp only [e_11_7, e_17_3, e_11_8] at hc
    linear_combination c585.trans k1 - hc

theorem privateBatchWrapper2_f72 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 17)) (a (.virt 81)) (a (.virt 82)) := by
  have c586 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c600 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c586, k2, ← c587, k2, ← c588] at e_11_10
  have e_11_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c589, ← c590, k2, ← c591] at e_11_11
  have e_11_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c598, ← c599, k2, ← c600] at e_11_12
  have e_17_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c592, ← c593, ← c594] at e_17_4
  have e_17_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c595, ← c596, ← c597] at e_17_5
  refine ⟨?_, ?_⟩
  · have hc := e_17_4
    simp only [e_11_11] at hc
    linear_combination c601.trans k1 - hc
  · have hc := e_11_12
    simp only [e_11_10, e_17_5, e_11_11] at hc
    linear_combination c602.trans k1 - hc

theorem privateBatchWrapper2_f73 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 18)) (a (.virt 83)) (a (.virt 84)) := by
  have c603 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (privateBatchWrapper2_copies19 a h).1
  have c609 := (privateBatchWrapper2_copies19 a h).2.1
  have c610 := (privateBatchWrapper2_copies19 a h).2.2.1
  have c611 := (privateBatchWrapper2_copies19 a h).2.2.2.1
  have c612 := (privateBatchWrapper2_copies19 a h).2.2.2.2.1
  have c613 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.1
  have c614 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.1
  have c615 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_11_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c603, k2, ← c604, k2, ← c605] at e_11_13
  have e_11_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c606, ← c607, k2, ← c608] at e_11_14
  have e_17_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c609, ← c610, ← c611] at e_17_6
  have e_17_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c612, ← c613, ← c614] at e_17_7
  have e_18_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c615, ← c616, k2, ← c617] at e_18_0
  refine ⟨?_, ?_⟩
  · have hc := e_17_6
    simp only [e_11_14] at hc
    linear_combination c618.trans k1 - hc
  · have hc := e_18_0
    simp only [e_11_13, e_17_7, e_11_14] at hc
    linear_combination c619.trans k1 - hc

theorem privateBatchWrapper2_f74 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 35) = band (a (.virt 77)) (a (.virt 79)) := by
  have c620 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c620, ← c621, ← c622] at e_17_8
  have hr := e_17_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f75 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 39) = band (a (.virt 81)) (a (.virt 83)) := by
  have c623 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c623, ← c624, ← c625] at e_17_9
  have hr := e_17_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f76 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 43) = band (a (.wire 17 35)) (a (.wire 17 39)) := by
  have c626 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c626, ← c627, ← c628] at e_17_10
  have hr := e_17_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f77 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 47) = band (a (.wire 9 59)) (a (.wire 17 43)) := by
  have c629 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c629, ← c630, ← c631] at e_17_11
  have hr := e_17_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f78 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 47) = a (.virt 11) := by
  have c632 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c632

theorem privateBatchWrapper2_f79 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 31) = bnot (a (.wire 4 15)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have e_5_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c148, k2, ← c149, k2, ← c150] at e_5_7
  have hr := e_5_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f80 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 19 12) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 0 ∧ a (.wire 19 13) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 1 ∧ a (.wire 19 14) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 2 ∧ a (.wire 19 15) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 3) := by
  have c633 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (privateBatchWrapper2_copies20 a h).1
  have c641 := (privateBatchWrapper2_copies20 a h).2.1
  have c642 := (privateBatchWrapper2_copies20 a h).2.2.1
  have c643 := (privateBatchWrapper2_copies20 a h).2.2.2.1
  have c644 := (privateBatchWrapper2_copies20 a h).2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 19 = addBlock (fun _ => 0) [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39), 1, 0, 0, 0] :=
    poseidon2In_first c633.symm c634.symm c635.symm c636.symm (c637.symm.trans k2) (c638.symm.trans k1) (c639.symm.trans k1) (c640.symm.trans k1) (c641.symm.trans k1) (c642.symm.trans k1) (c643.symm.trans k1) (c644.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 19) rfl rfl hin0
  have hpad : pad10 [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] =
      [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f81 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 20 12) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 0 ∧ a (.wire 20 13) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 1 ∧ a (.wire 20 14) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 2 ∧ a (.wire 20 15) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 3) := by
  have c645 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.1
  have c646 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.1
  have c647 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 20 = addBlock (fun _ => 0) [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15), 1, 0, 0, 0] :=
    poseidon2In_first c645.symm c646.symm c647.symm c648.symm (c649.symm.trans k2) (c650.symm.trans k1) (c651.symm.trans k1) (c652.symm.trans k1) (c653.symm.trans k1) (c654.symm.trans k1) (c655.symm.trans k1) (c656.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 20) rfl rfl hin0
  have hpad : pad10 [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] =
      [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f82 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 11) = bselect (a (.wire 3 31)) (a (.wire 20 12)) (a (.virt 1)) := by
  have c657 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c657, ← c658, ← c659] at e_18_1
  have e_18_2 := arithEq_of_rows h (i := 2) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c660, ← c661, ← c662] at e_18_2
  have hr := e_18_2
  simp only [e_18_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f83 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 19) = bselect (a (.wire 3 31)) (a (.wire 20 13)) (a (.virt 2)) := by
  have c663 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_3 := arithEq_of_rows h (i := 3) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c663, ← c664, ← c665] at e_18_3
  have e_18_4 := arithEq_of_rows h (i := 4) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c666, ← c667, ← c668] at e_18_4
  have hr := e_18_4
  simp only [e_18_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f84 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 27) = bselect (a (.wire 3 31)) (a (.wire 20 14)) (a (.virt 3)) := by
  have c669 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (privateBatchWrapper2_copies21 a h).1
  have c673 := (privateBatchWrapper2_copies21 a h).2.1
  have c674 := (privateBatchWrapper2_copies21 a h).2.2.1
  have e_18_5 := arithEq_of_rows h (i := 5) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c669, ← c670, ← c671] at e_18_5
  have e_18_6 := arithEq_of_rows h (i := 6) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c672, ← c673, ← c674] at e_18_6
  have hr := e_18_6
  simp only [e_18_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f85 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 35) = bselect (a (.wire 3 31)) (a (.wire 20 15)) (a (.virt 4)) := by
  have c675 := (privateBatchWrapper2_copies21 a h).2.2.2.1
  have c676 := (privateBatchWrapper2_copies21 a h).2.2.2.2.1
  have c677 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.1
  have c678 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.1
  have c679 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.1
  have e_18_7 := arithEq_of_rows h (i := 7) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c675, ← c676, ← c677] at e_18_7
  have e_18_8 := arithEq_of_rows h (i := 8) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c678, ← c679, ← c680] at e_18_8
  have hr := e_18_8
  simp only [e_18_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f86 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 21 12) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 0 ∧ a (.wire 21 13) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 1 ∧ a (.wire 21 14) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 2 ∧ a (.wire 21 15) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 3) := by
  have c681 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 21 = addBlock (fun _ => 0) [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43), 1, 0, 0, 0] :=
    poseidon2In_first c681.symm c682.symm c683.symm c684.symm (c685.symm.trans k2) (c686.symm.trans k1) (c687.symm.trans k1) (c688.symm.trans k1) (c689.symm.trans k1) (c690.symm.trans k1) (c691.symm.trans k1) (c692.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 21) rfl rfl hin0
  have hpad : pad10 [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] =
      [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f87 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 22 12) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 0 ∧ a (.wire 22 13) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 1 ∧ a (.wire 22 14) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 2 ∧ a (.wire 22 15) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 3) := by
  have c693 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (privateBatchWrapper2_copies22 a h).1
  have hk := privateBatchWrapper2_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 22 = addBlock (fun _ => 0) [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15), 1, 0, 0, 0] :=
    poseidon2In_first c693.symm c694.symm c695.symm c696.symm (c697.symm.trans k2) (c698.symm.trans k1) (c699.symm.trans k1) (c700.symm.trans k1) (c701.symm.trans k1) (c702.symm.trans k1) (c703.symm.trans k1) (c704.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 22) rfl rfl hin0
  have hpad : pad10 [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] =
      [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f88 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 43) = bselect (a (.wire 4 15)) (a (.wire 22 12)) (a (.virt 15)) := by
  have c705 := (privateBatchWrapper2_copies22 a h).2.1
  have c706 := (privateBatchWrapper2_copies22 a h).2.2.1
  have c707 := (privateBatchWrapper2_copies22 a h).2.2.2.1
  have c708 := (privateBatchWrapper2_copies22 a h).2.2.2.2.1
  have c709 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.1
  have c710 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.1
  have e_18_9 := arithEq_of_rows h (i := 9) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c705, ← c706, ← c707] at e_18_9
  have e_18_10 := arithEq_of_rows h (i := 10) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c708, ← c709, ← c710] at e_18_10
  have hr := e_18_10
  simp only [e_18_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f89 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 51) = bselect (a (.wire 4 15)) (a (.wire 22 13)) (a (.virt 16)) := by
  have c711 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_11 := arithEq_of_rows h (i := 11) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c711, ← c712, ← c713] at e_18_11
  have e_18_12 := arithEq_of_rows h (i := 12) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c714, ← c715, ← c716] at e_18_12
  have hr := e_18_12
  simp only [e_18_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f90 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 59) = bselect (a (.wire 4 15)) (a (.wire 22 14)) (a (.virt 17)) := by
  have c717 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_13 := arithEq_of_rows h (i := 13) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c717, ← c718, ← c719] at e_18_13
  have e_18_14 := arithEq_of_rows h (i := 14) privateBatchWrapper2_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c720, ← c721, ← c722] at e_18_14
  have hr := e_18_14
  simp only [e_18_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f91 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 23 7) = bselect (a (.wire 4 15)) (a (.wire 22 15)) (a (.virt 18)) := by
  have c723 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_23_0 := arithEq_of_rows h (i := 0) privateBatchWrapper2_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_0
  simp only [← c723, ← c724, ← c725] at e_23_0
  have e_23_1 := arithEq_of_rows h (i := 1) privateBatchWrapper2_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_1
  simp only [← c726, ← c727, ← c728] at e_23_1
  have hr := e_23_1
  simp only [e_23_0] at hr
  simp only [bselect]
  linear_combination hr

end Plonky2Spec.Generated
