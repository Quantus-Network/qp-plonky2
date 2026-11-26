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

theorem privateBatchWrapper8_f644 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 11) = bselect (a (.wire 8 35)) (a (.wire 132 13)) (a (.virt 59)) := by
  have c5325 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5326 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5327 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5328 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5329 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5330 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_1 := arithEq_of_rows h (row := 133) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_1
  simp only [← c5325, ← c5326, ← c5327] at e_133_1
  have e_133_2 := arithEq_of_rows h (row := 133) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_2
  simp only [← c5328, ← c5329, ← c5330] at e_133_2
  have hr := e_133_2
  simp only [e_133_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f645 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 19) = bselect (a (.wire 8 35)) (a (.wire 132 14)) (a (.virt 60)) := by
  have c5331 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5332 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5333 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5334 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5335 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5336 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_3 := arithEq_of_rows h (row := 133) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_3
  simp only [← c5331, ← c5332, ← c5333] at e_133_3
  have e_133_4 := arithEq_of_rows h (row := 133) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_4
  simp only [← c5334, ← c5335, ← c5336] at e_133_4
  have hr := e_133_4
  simp only [e_133_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f646 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 27) = bselect (a (.wire 8 35)) (a (.wire 132 15)) (a (.virt 61)) := by
  have c5337 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5338 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5339 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5340 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5341 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5342 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_5 := arithEq_of_rows h (row := 133) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_5
  simp only [← c5337, ← c5338, ← c5339] at e_133_5
  have e_133_6 := arithEq_of_rows h (row := 133) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_6
  simp only [← c5340, ← c5341, ← c5342] at e_133_6
  have hr := e_133_6
  simp only [e_133_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f647 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 134 12) = spongeHash perm [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278)] 0 ∧ a (.wire 134 13) = spongeHash perm [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278)] 1 ∧ a (.wire 134 14) = spongeHash perm [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278)] 2 ∧ a (.wire 134 15) = spongeHash perm [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278)] 3) := by
  have c5343 := (privateBatchWrapper8_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5344 := (privateBatchWrapper8_copies167 a h).1
  have c5345 := (privateBatchWrapper8_copies167 a h).2.1
  have c5346 := (privateBatchWrapper8_copies167 a h).2.2.1
  have c5347 := (privateBatchWrapper8_copies167 a h).2.2.2.1
  have c5348 := (privateBatchWrapper8_copies167 a h).2.2.2.2.1
  have c5349 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.1
  have c5350 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.1
  have c5351 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.1
  have c5352 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.1
  have c5353 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.1
  have c5354 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 134 = addBlock (fun _ => 0) [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), 1, 0, 0, 0] :=
    poseidon2In_first c5343.symm c5344.symm c5345.symm c5346.symm (c5347.symm.trans k3) (c5348.symm.trans k2) (c5349.symm.trans k2) (c5350.symm.trans k2) (c5351.symm.trans k2) (c5352.symm.trans k2) (c5353.symm.trans k2) (c5354.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 134) rfl rfl hin0
  have hpad : pad10 [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278)] =
      [a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f648 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 135 12) = spongeHash perm [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15)] 0 ∧ a (.wire 135 13) = spongeHash perm [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15)] 1 ∧ a (.wire 135 14) = spongeHash perm [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15)] 2 ∧ a (.wire 135 15) = spongeHash perm [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15)] 3) := by
  have c5355 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5356 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5357 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5358 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5359 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5360 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5361 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5362 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5363 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5364 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5365 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5366 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 135 = addBlock (fun _ => 0) [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15), 1, 0, 0, 0] :=
    poseidon2In_first c5355.symm c5356.symm c5357.symm c5358.symm (c5359.symm.trans k3) (c5360.symm.trans k2) (c5361.symm.trans k2) (c5362.symm.trans k2) (c5363.symm.trans k2) (c5364.symm.trans k2) (c5365.symm.trans k2) (c5366.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 135) rfl rfl hin0
  have hpad : pad10 [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15)] =
      [a (.wire 134 12), a (.wire 134 13), a (.wire 134 14), a (.wire 134 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f649 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 35) = bselect (a (.wire 10 19)) (a (.wire 135 12)) (a (.virt 69)) := by
  have c5367 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5368 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5369 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5370 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5371 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5372 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_7 := arithEq_of_rows h (row := 133) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_7
  simp only [← c5367, ← c5368, ← c5369] at e_133_7
  have e_133_8 := arithEq_of_rows h (row := 133) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_8
  simp only [← c5370, ← c5371, ← c5372] at e_133_8
  have hr := e_133_8
  simp only [e_133_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f650 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 43) = bselect (a (.wire 10 19)) (a (.wire 135 13)) (a (.virt 70)) := by
  have c5373 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5374 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5375 := (privateBatchWrapper8_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5376 := (privateBatchWrapper8_copies168 a h).1
  have c5377 := (privateBatchWrapper8_copies168 a h).2.1
  have c5378 := (privateBatchWrapper8_copies168 a h).2.2.1
  have e_133_9 := arithEq_of_rows h (row := 133) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_9
  simp only [← c5373, ← c5374, ← c5375] at e_133_9
  have e_133_10 := arithEq_of_rows h (row := 133) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_10
  simp only [← c5376, ← c5377, ← c5378] at e_133_10
  have hr := e_133_10
  simp only [e_133_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f651 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 51) = bselect (a (.wire 10 19)) (a (.wire 135 14)) (a (.virt 71)) := by
  have c5379 := (privateBatchWrapper8_copies168 a h).2.2.2.1
  have c5380 := (privateBatchWrapper8_copies168 a h).2.2.2.2.1
  have c5381 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.1
  have c5382 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.1
  have c5383 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.1
  have c5384 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.1
  have e_133_11 := arithEq_of_rows h (row := 133) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_11
  simp only [← c5379, ← c5380, ← c5381] at e_133_11
  have e_133_12 := arithEq_of_rows h (row := 133) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_12
  simp only [← c5382, ← c5383, ← c5384] at e_133_12
  have hr := e_133_12
  simp only [e_133_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f652 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 133 59) = bselect (a (.wire 10 19)) (a (.wire 135 15)) (a (.virt 72)) := by
  have c5385 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.1
  have c5386 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_13 := arithEq_of_rows h (row := 133) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_13
  simp only [← c5385, ← c5386, ← c5387] at e_133_13
  have e_133_14 := arithEq_of_rows h (row := 133) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_14
  simp only [← c5388, ← c5389, ← c5390] at e_133_14
  have hr := e_133_14
  simp only [e_133_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f653 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 136 12) = spongeHash perm [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282)] 0 ∧ a (.wire 136 13) = spongeHash perm [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282)] 1 ∧ a (.wire 136 14) = spongeHash perm [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282)] 2 ∧ a (.wire 136 15) = spongeHash perm [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282)] 3) := by
  have c5391 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5392 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5393 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5394 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5395 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5396 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5397 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5398 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5399 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5400 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5401 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5402 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 136 = addBlock (fun _ => 0) [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), 1, 0, 0, 0] :=
    poseidon2In_first c5391.symm c5392.symm c5393.symm c5394.symm (c5395.symm.trans k3) (c5396.symm.trans k2) (c5397.symm.trans k2) (c5398.symm.trans k2) (c5399.symm.trans k2) (c5400.symm.trans k2) (c5401.symm.trans k2) (c5402.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 136) rfl rfl hin0
  have hpad : pad10 [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282)] =
      [a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f654 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper8 p) a) :
    (a (.wire 137 12) = spongeHash perm [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15)] 0 ∧ a (.wire 137 13) = spongeHash perm [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15)] 1 ∧ a (.wire 137 14) = spongeHash perm [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15)] 2 ∧ a (.wire 137 15) = spongeHash perm [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15)] 3) := by
  have c5403 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5404 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5405 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5406 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5407 := (privateBatchWrapper8_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5408 := (privateBatchWrapper8_copies169 a h).1
  have c5409 := (privateBatchWrapper8_copies169 a h).2.1
  have c5410 := (privateBatchWrapper8_copies169 a h).2.2.1
  have c5411 := (privateBatchWrapper8_copies169 a h).2.2.2.1
  have c5412 := (privateBatchWrapper8_copies169 a h).2.2.2.2.1
  have c5413 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.1
  have c5414 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.1
  have hk := privateBatchWrapper8_consts a h
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31⟩ := hk.1.1.1.1
  have hin0 : poseidon2In a 137 = addBlock (fun _ => 0) [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15), 1, 0, 0, 0] :=
    poseidon2In_first c5403.symm c5404.symm c5405.symm c5406.symm (c5407.symm.trans k3) (c5408.symm.trans k2) (c5409.symm.trans k2) (c5410.symm.trans k2) (c5411.symm.trans k2) (c5412.symm.trans k2) (c5413.symm.trans k2) (c5414.symm.trans k2)
  have hout0 := poseidon2Row_absorb perm hp (row := 137) rfl rfl hin0
  have hpad : pad10 [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15)] =
      [a (.wire 136 12), a (.wire 136 13), a (.wire 136 14), a (.wire 136 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper8_f655 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 138 7) = bselect (a (.wire 12 3)) (a (.wire 137 12)) (a (.virt 80)) := by
  have c5415 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.1
  have c5416 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.1
  have c5417 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.1
  have c5418 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5419 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5420 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_0 := arithEq_of_rows h (row := 138) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_0
  simp only [← c5415, ← c5416, ← c5417] at e_138_0
  have e_138_1 := arithEq_of_rows h (row := 138) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_1
  simp only [← c5418, ← c5419, ← c5420] at e_138_1
  have hr := e_138_1
  simp only [e_138_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f656 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 138 15) = bselect (a (.wire 12 3)) (a (.wire 137 13)) (a (.virt 81)) := by
  have c5421 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5422 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5423 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5424 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5425 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5426 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_2 := arithEq_of_rows h (row := 138) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_2
  simp only [← c5421, ← c5422, ← c5423] at e_138_2
  have e_138_3 := arithEq_of_rows h (row := 138) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_3
  simp only [← c5424, ← c5425, ← c5426] at e_138_3
  have hr := e_138_3
  simp only [e_138_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f657 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 138 23) = bselect (a (.wire 12 3)) (a (.wire 137 14)) (a (.virt 82)) := by
  have c5427 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5428 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5429 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5430 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5431 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5432 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_4 := arithEq_of_rows h (row := 138) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_4
  simp only [← c5427, ← c5428, ← c5429] at e_138_4
  have e_138_5 := arithEq_of_rows h (row := 138) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_5
  simp only [← c5430, ← c5431, ← c5432] at e_138_5
  have hr := e_138_5
  simp only [e_138_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper8_f658 (a : Assignment p) (h : Satisfies (privateBatchWrapper8 p) a) :
    a (.wire 138 31) = bselect (a (.wire 12 3)) (a (.wire 137 15)) (a (.virt 83)) := by
  have c5433 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5434 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5435 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5436 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5437 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5438 := (privateBatchWrapper8_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_6 := arithEq_of_rows h (row := 138) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_6
  simp only [← c5433, ← c5434, ← c5435] at e_138_6
  have e_138_7 := arithEq_of_rows h (row := 138) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_7
  simp only [← c5436, ← c5437, ← c5438] at e_138_7
  have hr := e_138_7
  simp only [e_138_6] at hr
  simp only [bselect]
  linear_combination hr

end Plonky2Spec.Generated
