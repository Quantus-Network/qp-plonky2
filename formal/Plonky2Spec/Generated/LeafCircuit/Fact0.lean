/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.LeafCircuit`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge
import Plonky2Spec.Generated.LeafCircuit.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem leafCircuit_f0 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsBool (a (.virt 13)) := by
  have c0 := (leafCircuit_copies0 a h).1
  have c1 := (leafCircuit_copies0 a h).2.1
  have c2 := (leafCircuit_copies0 a h).2.2.1
  have c3 := (leafCircuit_copies0 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_0 := arithEq_of_rows h (i := 0) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, ← c1, ← c2] at e_0_0
  refine isBool_iff_assertBool.mpr ?_
  have hc := e_0_0
  linear_combination c3.trans k0 - hc

theorem leafCircuit_f1 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 1 12) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 0 ∧ a (.wire 1 13) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 1 ∧ a (.wire 1 14) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 2 ∧ a (.wire 1 15) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 3) := by
  have c4 := (leafCircuit_copies0 a h).2.2.2.2.1
  have c5 := (leafCircuit_copies0 a h).2.2.2.2.2.1
  have c6 := (leafCircuit_copies0 a h).2.2.2.2.2.2.1
  have c7 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hin0 : poseidon2In a 1 = addBlock (fun _ => 0) [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240), 1] :=
    poseidon2In_first c4.symm c5.symm c6.symm c7.symm c8.symm c9.symm c10.symm (c11.symm.trans k3) (c12.symm.trans k0) (c13.symm.trans k0) (c14.symm.trans k0) (c15.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 1) rfl rfl hin0
  have hpad : pad10 [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] =
      [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240), 1] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f2 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 2 12) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 0 ∧ a (.wire 2 13) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 1 ∧ a (.wire 2 14) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 2 ∧ a (.wire 2 15) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 3) := by
  have c16 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hin0 : poseidon2In a 2 = addBlock (fun _ => 0) [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15), 1, 0, 0, 0] :=
    poseidon2In_first c16.symm c17.symm c18.symm c19.symm (c20.symm.trans k3) (c21.symm.trans k0) (c22.symm.trans k0) (c23.symm.trans k0) (c24.symm.trans k0) (c25.symm.trans k0) (c26.symm.trans k0) (c27.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 2) rfl rfl hin0
  have hpad : pad10 [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] =
      [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f3 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 2 12) = a (.virt 233) := by
  have c28 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c28

theorem leafCircuit_f4 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 2 13) = a (.virt 234) := by
  have c29 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c29

theorem leafCircuit_f5 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 2 14) = a (.virt 235) := by
  have c30 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c30

theorem leafCircuit_f6 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 2 15) = a (.virt 236) := by
  have c31 := (leafCircuit_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c31

theorem leafCircuit_f7 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 5)) 32 := by
  have c32 := (leafCircuit_copies1 a h).1
  have c33 := (leafCircuit_copies1 a h).2.1
  have c34 := (leafCircuit_copies1 a h).2.2.1
  have c35 := (leafCircuit_copies1 a h).2.2.2.1
  have c36 := (leafCircuit_copies1 a h).2.2.2.2.1
  have c37 := (leafCircuit_copies1 a h).2.2.2.2.2.1
  have c38 := (leafCircuit_copies1 a h).2.2.2.2.2.2.1
  have c39 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.1
  have c42 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 3) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c32.trans k0
    · exact c33.trans k0
    · exact c34.trans k0
    · exact c35.trans k0
    · exact c36.trans k0
    · exact c37.trans k0
    · exact c38.trans k0
    · exact c39.trans k0
    · exact c40.trans k0
    · exact c41.trans k0
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
    )
  rwa [c59] at hr

theorem leafCircuit_f8 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 6)) 32 := by
  have c60 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (leafCircuit_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (leafCircuit_copies2 a h).1
  have c65 := (leafCircuit_copies2 a h).2.1
  have c66 := (leafCircuit_copies2 a h).2.2.1
  have c67 := (leafCircuit_copies2 a h).2.2.2.1
  have c68 := (leafCircuit_copies2 a h).2.2.2.2.1
  have c69 := (leafCircuit_copies2 a h).2.2.2.2.2.1
  have c70 := (leafCircuit_copies2 a h).2.2.2.2.2.2.1
  have c71 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 4) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
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
    )
  rwa [c87] at hr

theorem leafCircuit_f9 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 0)) 32 := by
  have c88 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (leafCircuit_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (leafCircuit_copies3 a h).1
  have c97 := (leafCircuit_copies3 a h).2.1
  have c98 := (leafCircuit_copies3 a h).2.2.1
  have c99 := (leafCircuit_copies3 a h).2.2.2.1
  have c100 := (leafCircuit_copies3 a h).2.2.2.2.1
  have c101 := (leafCircuit_copies3 a h).2.2.2.2.2.1
  have c102 := (leafCircuit_copies3 a h).2.2.2.2.2.2.1
  have c103 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 5) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c88.trans k0
    · exact c89.trans k0
    · exact c90.trans k0
    · exact c91.trans k0
    · exact c92.trans k0
    · exact c93.trans k0
    · exact c94.trans k0
    · exact c95.trans k0
    · exact c96.trans k0
    · exact c97.trans k0
    · exact c98.trans k0
    · exact c99.trans k0
    · exact c100.trans k0
    · exact c101.trans k0
    · exact c102.trans k0
    · exact c103.trans k0
    · exact c104.trans k0
    · exact c105.trans k0
    · exact c106.trans k0
    · exact c107.trans k0
    · exact c108.trans k0
    · exact c109.trans k0
    · exact c110.trans k0
    · exact c111.trans k0
    · exact c112.trans k0
    · exact c113.trans k0
    · exact c114.trans k0
    )
  rwa [c115] at hr

theorem leafCircuit_f10 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 7)) 32 := by
  have c116 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c127 := (leafCircuit_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (leafCircuit_copies4 a h).1
  have c129 := (leafCircuit_copies4 a h).2.1
  have c130 := (leafCircuit_copies4 a h).2.2.1
  have c131 := (leafCircuit_copies4 a h).2.2.2.1
  have c132 := (leafCircuit_copies4 a h).2.2.2.2.1
  have c133 := (leafCircuit_copies4 a h).2.2.2.2.2.1
  have c134 := (leafCircuit_copies4 a h).2.2.2.2.2.2.1
  have c135 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 6) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c116.trans k0
    · exact c117.trans k0
    · exact c118.trans k0
    · exact c119.trans k0
    · exact c120.trans k0
    · exact c121.trans k0
    · exact c122.trans k0
    · exact c123.trans k0
    · exact c124.trans k0
    · exact c125.trans k0
    · exact c126.trans k0
    · exact c127.trans k0
    · exact c128.trans k0
    · exact c129.trans k0
    · exact c130.trans k0
    · exact c131.trans k0
    · exact c132.trans k0
    · exact c133.trans k0
    · exact c134.trans k0
    · exact c135.trans k0
    · exact c136.trans k0
    · exact c137.trans k0
    · exact c138.trans k0
    · exact c139.trans k0
    · exact c140.trans k0
    · exact c141.trans k0
    · exact c142.trans k0
    )
  rwa [c143] at hr

theorem leafCircuit_f11 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 9 12) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 0 ∧ a (.wire 9 13) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 1 ∧ a (.wire 9 14) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 2 ∧ a (.wire 9 15) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 3) := by
  have c144 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c145 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c148 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c151 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (leafCircuit_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (leafCircuit_copies5 a h).1
  have c161 := (leafCircuit_copies5 a h).2.1
  have c162 := (leafCircuit_copies5 a h).2.2.1
  have c163 := (leafCircuit_copies5 a h).2.2.2.1
  have c164 := (leafCircuit_copies5 a h).2.2.2.2.1
  have c165 := (leafCircuit_copies5 a h).2.2.2.2.2.1
  have c166 := (leafCircuit_copies5 a h).2.2.2.2.2.2.1
  have c167 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_0 := arithEq_of_rows h (i := 0) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c156, ← c157, k3, ← c158, k3] at e_8_0
  have hin0 : poseidon2In a 7 = addBlock (fun _ => 0) [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] :=
    poseidon2In_first c144.symm c145.symm c146.symm c147.symm c148.symm c149.symm c150.symm c151.symm (c152.symm.trans k0) (c153.symm.trans k0) (c154.symm.trans k0) (c155.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 7) rfl rfl hin0
  have hin1 : poseidon2In a 9 = addBlock (poseidon2Out a 7) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c159 + e_8_0) (c160.symm.trans (add_zero _).symm) (c161.symm.trans (add_zero _).symm) (c162.symm.trans (add_zero _).symm) (c163.symm.trans (add_zero _).symm) (c164.symm.trans (add_zero _).symm) (c165.symm.trans (add_zero _).symm) (c166.symm.trans (add_zero _).symm) c167.symm c168.symm c169.symm c170.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 9) rfl rfl hin1
  have hpad : pad10 [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] =
      [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ []) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f12 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 10 1), a (.wire 10 2), a (.wire 10 3), a (.wire 10 4), a (.wire 10 5)] := by
  have c171 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (leafCircuit_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (leafCircuit_copies6 a h).1
  have c193 := (leafCircuit_copies6 a h).2.1
  have c194 := (leafCircuit_copies6 a h).2.2.1
  have c195 := (leafCircuit_copies6 a h).2.2.2.1
  have c196 := (leafCircuit_copies6 a h).2.2.2.2.1
  have c197 := (leafCircuit_copies6 a h).2.2.2.2.2.1
  have c198 := (leafCircuit_copies6 a h).2.2.2.2.2.2.1
  have c199 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (leafCircuit_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (leafCircuit_copies7 a h).1
  have c225 := (leafCircuit_copies7 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 10) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c171.trans k0
    · exact c172.trans k0
    · exact c173.trans k0
    · exact c174.trans k0
    · exact c175.trans k0
    · exact c176.trans k0
    · exact c177.trans k0
    · exact c178.trans k0
    · exact c179.trans k0
    · exact c180.trans k0
    · exact c181.trans k0
    · exact c182.trans k0
    · exact c183.trans k0
    · exact c184.trans k0
    · exact c185.trans k0
    · exact c186.trans k0
    · exact c187.trans k0
    · exact c188.trans k0
    · exact c189.trans k0
    · exact c190.trans k0
    · exact c191.trans k0
    · exact c192.trans k0
    · exact c193.trans k0
    · exact c194.trans k0
    · exact c195.trans k0
    · exact c196.trans k0
    · exact c197.trans k0
    · exact c198.trans k0
    · exact c199.trans k0
    · exact c200.trans k0
    · exact c201.trans k0
    · exact c202.trans k0
    · exact c203.trans k0
    · exact c204.trans k0
    · exact c205.trans k0
    · exact c206.trans k0
    · exact c207.trans k0
    · exact c208.trans k0
    · exact c209.trans k0
    · exact c210.trans k0
    · exact c211.trans k0
    · exact c212.trans k0
    · exact c213.trans k0
    · exact c214.trans k0
    · exact c215.trans k0
    · exact c216.trans k0
    · exact c217.trans k0
    · exact c218.trans k0
    · exact c219.trans k0
    · exact c220.trans k0
    · exact c221.trans k0
    · exact c222.trans k0
    · exact c223.trans k0
    · exact c224.trans k0
    )
  rwa [c225] at hr

theorem leafCircuit_f13 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f14 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 10 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f15 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0, k3]
  ring

theorem leafCircuit_f16 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bor (a (.virt 14)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f17 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 5) = a (.virt 292) * a (.wire 10 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f18 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 3) = a (.virt 294) * a (.wire 10 5) := by
  have c226 := (leafCircuit_copies7 a h).2.2.1
  have c227 := (leafCircuit_copies7 a h).2.2.2.1
  have c228 := (leafCircuit_copies7 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_11_0 := arithEq_of_rows h (i := 0) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c226, k5, ← c227, ← c228, k5] at e_11_0
  have hr := e_11_0
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f19 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 7) = a (.virt 292) + a (.wire 10 5) := by
  have c229 := (leafCircuit_copies7 a h).2.2.2.2.2.1
  have c230 := (leafCircuit_copies7 a h).2.2.2.2.2.2.1
  have c231 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_1 := arithEq_of_rows h (i := 1) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c229, k3, ← c230, k3, ← c231] at e_8_1
  have hr := e_8_1
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f20 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 7) = a (.wire 8 7) - a (.wire 11 3) := by
  have c232 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_1 := arithEq_of_rows h (i := 1) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_1
  simp only [← c232, ← c233, k3, ← c234] at e_0_1
  have hr := e_0_1
  linear_combination hr

theorem leafCircuit_f21 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 11) = bnot (a (.wire 0 7)) := by
  have c235 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_2 := arithEq_of_rows h (i := 2) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_2
  simp only [← c235, k3, ← c236, k3, ← c237] at e_0_2
  have hr := e_0_2
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f22 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 11) = band (a (.virt 292)) (a (.wire 0 11)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f23 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f24 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 4) = band (a (.virt 292)) (a (.wire 10 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f25 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 7) = band (a (.wire 10 4)) (a (.wire 0 11)) := by
  have c238 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_1 := arithEq_of_rows h (i := 1) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c238, ← c239, ← c240] at e_11_1
  have hr := e_11_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f26 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 7) = bor (a (.virt 14)) (a (.wire 11 7)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f27 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 10 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f28 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f29 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 4) = a (.virt 14) + a (.wire 10 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f30 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 4) = a (.wire 10 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f31 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 15) = bnot (a (.wire 10 4)) := by
  have c241 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_3 := arithEq_of_rows h (i := 3) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_3
  simp only [← c241, k3, ← c242, k3, ← c243] at e_0_3
  have hr := e_0_3
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f32 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 11) = band (a (.wire 0 11)) (a (.wire 0 15)) := by
  have c244 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_2 := arithEq_of_rows h (i := 2) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c244, ← c245, ← c246] at e_11_2
  have hr := e_11_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f33 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f34 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 3) = band (a (.virt 292)) (a (.wire 10 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f35 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 15) = band (a (.wire 10 3)) (a (.wire 11 11)) := by
  have c247 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_3 := arithEq_of_rows h (i := 3) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c247, ← c248, ← c249] at e_11_3
  have hr := e_11_3
  simp only [band]
  linear_combination hr

theorem leafCircuit_f36 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 11) = bor (a (.wire 11 7)) (a (.wire 11 15)) := by
  have c250 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (leafCircuit_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_2 := arithEq_of_rows h (i := 2) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c253, ← c254, k3, ← c255] at e_8_2
  have e_12_0 := arithEq_of_rows h (i := 0) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c250, ← c251, ← c252] at e_12_0
  have hr := e_8_2
  simp only [e_12_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f37 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 10 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f38 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f39 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 3) = a (.virt 14) + a (.wire 10 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f40 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 3) = a (.wire 10 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f41 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 19) = bnot (a (.wire 10 3)) := by
  have c256 := (leafCircuit_copies8 a h).1
  have c257 := (leafCircuit_copies8 a h).2.1
  have c258 := (leafCircuit_copies8 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_4 := arithEq_of_rows h (i := 4) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_4
  simp only [← c256, k3, ← c257, k3, ← c258] at e_0_4
  have hr := e_0_4
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f42 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 19) = band (a (.wire 11 11)) (a (.wire 0 19)) := by
  have c259 := (leafCircuit_copies8 a h).2.2.2.1
  have c260 := (leafCircuit_copies8 a h).2.2.2.2.1
  have c261 := (leafCircuit_copies8 a h).2.2.2.2.2.1
  have e_11_4 := arithEq_of_rows h (i := 4) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c259, ← c260, ← c261] at e_11_4
  have hr := e_11_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f43 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f44 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 2) = band (a (.virt 292)) (a (.wire 10 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f45 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 23) = band (a (.wire 10 2)) (a (.wire 11 19)) := by
  have c262 := (leafCircuit_copies8 a h).2.2.2.2.2.2.1
  have c263 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.1
  have e_11_5 := arithEq_of_rows h (i := 5) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c262, ← c263, ← c264] at e_11_5
  have hr := e_11_5
  simp only [band]
  linear_combination hr

theorem leafCircuit_f46 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 15) = bor (a (.wire 8 11)) (a (.wire 11 23)) := by
  have c265 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_3 := arithEq_of_rows h (i := 3) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c268, ← c269, k3, ← c270] at e_8_3
  have e_12_1 := arithEq_of_rows h (i := 1) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c265, ← c266, ← c267] at e_12_1
  have hr := e_8_3
  simp only [e_12_1] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f47 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 10 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f48 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f49 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 2) = a (.virt 14) + a (.wire 10 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f50 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 2) = a (.wire 10 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f51 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 23) = bnot (a (.wire 10 2)) := by
  have c271 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_5 := arithEq_of_rows h (i := 5) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_5
  simp only [← c271, k3, ← c272, k3, ← c273] at e_0_5
  have hr := e_0_5
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f52 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 27) = band (a (.wire 11 19)) (a (.wire 0 23)) := by
  have c274 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_6 := arithEq_of_rows h (i := 6) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c274, ← c275, ← c276] at e_11_6
  have hr := e_11_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f53 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f54 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 1) = band (a (.virt 292)) (a (.wire 10 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f55 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 31) = band (a (.wire 10 1)) (a (.wire 11 27)) := by
  have c277 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c278 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_7 := arithEq_of_rows h (i := 7) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c277, ← c278, ← c279] at e_11_7
  have hr := e_11_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f56 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 19) = bor (a (.wire 8 15)) (a (.wire 11 31)) := by
  have c280 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_4 := arithEq_of_rows h (i := 4) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c283, ← c284, k3, ← c285] at e_8_4
  have e_12_2 := arithEq_of_rows h (i := 2) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c280, ← c281, ← c282] at e_12_2
  have hr := e_8_4
  simp only [e_12_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f57 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 10 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f58 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f59 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 1) = a (.virt 14) + a (.wire 10 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f60 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 10 1) = a (.wire 10 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f61 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 27) = bnot (a (.wire 10 1)) := by
  have c286 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (leafCircuit_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (leafCircuit_copies9 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_6 := arithEq_of_rows h (i := 6) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_6
  simp only [← c286, k3, ← c287, k3, ← c288] at e_0_6
  have hr := e_0_6
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f62 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 35) = band (a (.wire 11 27)) (a (.wire 0 27)) := by
  have c289 := (leafCircuit_copies9 a h).2.1
  have c290 := (leafCircuit_copies9 a h).2.2.1
  have c291 := (leafCircuit_copies9 a h).2.2.2.1
  have e_11_8 := arithEq_of_rows h (i := 8) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c289, ← c290, ← c291] at e_11_8
  have hr := e_11_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f63 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 19) = a (.virt 14) := by
  have c292 := (leafCircuit_copies9 a h).2.2.2.2.1
  exact c292

theorem leafCircuit_f64 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 13 1), a (.wire 13 2), a (.wire 13 3), a (.wire 13 4), a (.wire 13 5)] := by
  have c293 := (leafCircuit_copies9 a h).2.2.2.2.2.1
  have c294 := (leafCircuit_copies9 a h).2.2.2.2.2.2.1
  have c295 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (leafCircuit_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (leafCircuit_copies10 a h).1
  have c321 := (leafCircuit_copies10 a h).2.1
  have c322 := (leafCircuit_copies10 a h).2.2.1
  have c323 := (leafCircuit_copies10 a h).2.2.2.1
  have c324 := (leafCircuit_copies10 a h).2.2.2.2.1
  have c325 := (leafCircuit_copies10 a h).2.2.2.2.2.1
  have c326 := (leafCircuit_copies10 a h).2.2.2.2.2.2.1
  have c327 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 13) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c293.trans k0
    · exact c294.trans k0
    · exact c295.trans k0
    · exact c296.trans k0
    · exact c297.trans k0
    · exact c298.trans k0
    · exact c299.trans k0
    · exact c300.trans k0
    · exact c301.trans k0
    · exact c302.trans k0
    · exact c303.trans k0
    · exact c304.trans k0
    · exact c305.trans k0
    · exact c306.trans k0
    · exact c307.trans k0
    · exact c308.trans k0
    · exact c309.trans k0
    · exact c310.trans k0
    · exact c311.trans k0
    · exact c312.trans k0
    · exact c313.trans k0
    · exact c314.trans k0
    · exact c315.trans k0
    · exact c316.trans k0
    · exact c317.trans k0
    · exact c318.trans k0
    · exact c319.trans k0
    · exact c320.trans k0
    · exact c321.trans k0
    · exact c322.trans k0
    · exact c323.trans k0
    · exact c324.trans k0
    · exact c325.trans k0
    · exact c326.trans k0
    · exact c327.trans k0
    · exact c328.trans k0
    · exact c329.trans k0
    · exact c330.trans k0
    · exact c331.trans k0
    · exact c332.trans k0
    · exact c333.trans k0
    · exact c334.trans k0
    · exact c335.trans k0
    · exact c336.trans k0
    · exact c337.trans k0
    · exact c338.trans k0
    · exact c339.trans k0
    · exact c340.trans k0
    · exact c341.trans k0
    · exact c342.trans k0
    · exact c343.trans k0
    · exact c344.trans k0
    · exact c345.trans k0
    · exact c346.trans k0
    )
  rwa [c347] at hr

theorem leafCircuit_f65 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f66 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 5) = band (a (.virt 292)) (a (.wire 13 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f67 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 5) = band (a (.wire 13 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f68 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 5) = bor (a (.virt 14)) (a (.wire 13 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f69 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 13 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f70 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f71 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 5) = a (.virt 14) + a (.wire 13 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f72 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 5) = a (.wire 13 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f73 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 31) = bnot (a (.wire 13 5)) := by
  have c348 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_7 := arithEq_of_rows h (i := 7) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_7
  simp only [← c348, k3, ← c349, k3, ← c350] at e_0_7
  have hr := e_0_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f74 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 31) = band (a (.virt 292)) (a (.wire 0 31)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f75 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f76 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 4) = band (a (.virt 292)) (a (.wire 13 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f77 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 39) = band (a (.wire 13 4)) (a (.wire 0 31)) := by
  have c351 := (leafCircuit_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (leafCircuit_copies11 a h).1
  have c353 := (leafCircuit_copies11 a h).2.1
  have e_11_9 := arithEq_of_rows h (i := 9) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c351, ← c352, ← c353] at e_11_9
  have hr := e_11_9
  simp only [band]
  linear_combination hr

theorem leafCircuit_f78 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 23) = bor (a (.wire 13 5)) (a (.wire 11 39)) := by
  have c354 := (leafCircuit_copies11 a h).2.2.1
  have c355 := (leafCircuit_copies11 a h).2.2.2.1
  have c356 := (leafCircuit_copies11 a h).2.2.2.2.1
  have c357 := (leafCircuit_copies11 a h).2.2.2.2.2.1
  have c358 := (leafCircuit_copies11 a h).2.2.2.2.2.2.1
  have c359 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_5 := arithEq_of_rows h (i := 5) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c357, ← c358, k3, ← c359] at e_8_5
  have e_12_3 := arithEq_of_rows h (i := 3) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c354, ← c355, ← c356] at e_12_3
  have hr := e_8_5
  simp only [e_12_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f79 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 13 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f80 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f81 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 4) = a (.virt 14) + a (.wire 13 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f82 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 4) = a (.wire 13 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f83 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 35) = bnot (a (.wire 13 4)) := by
  have c360 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have c362 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_8 := arithEq_of_rows h (i := 8) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_8
  simp only [← c360, k3, ← c361, k3, ← c362] at e_0_8
  have hr := e_0_8
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f84 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 43) = band (a (.wire 0 31)) (a (.wire 0 35)) := by
  have c363 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_10 := arithEq_of_rows h (i := 10) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c363, ← c364, ← c365] at e_11_10
  have hr := e_11_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f85 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f86 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 3) = band (a (.virt 292)) (a (.wire 13 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f87 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 47) = band (a (.wire 13 3)) (a (.wire 11 43)) := by
  have c366 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_11 := arithEq_of_rows h (i := 11) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c366, ← c367, ← c368] at e_11_11
  have hr := e_11_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f88 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 27) = bor (a (.wire 8 23)) (a (.wire 11 47)) := by
  have c369 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_6 := arithEq_of_rows h (i := 6) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c372, ← c373, k3, ← c374] at e_8_6
  have e_12_4 := arithEq_of_rows h (i := 4) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c369, ← c370, ← c371] at e_12_4
  have hr := e_8_6
  simp only [e_12_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f89 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 13 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f90 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f91 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 3) = a (.virt 14) + a (.wire 13 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f92 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 3) = a (.wire 13 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f93 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 39) = bnot (a (.wire 13 3)) := by
  have c375 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_9 := arithEq_of_rows h (i := 9) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_9
  simp only [← c375, k3, ← c376, k3, ← c377] at e_0_9
  have hr := e_0_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f94 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 51) = band (a (.wire 11 43)) (a (.wire 0 39)) := by
  have c378 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_12 := arithEq_of_rows h (i := 12) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c378, ← c379, ← c380] at e_11_12
  have hr := e_11_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f95 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f96 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 2) = band (a (.virt 292)) (a (.wire 13 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f97 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 55) = band (a (.wire 13 2)) (a (.wire 11 51)) := by
  have c381 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (leafCircuit_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_11_13 := arithEq_of_rows h (i := 13) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c381, ← c382, ← c383] at e_11_13
  have hr := e_11_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f98 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 31) = bor (a (.wire 8 27)) (a (.wire 11 55)) := by
  have c384 := (leafCircuit_copies12 a h).1
  have c385 := (leafCircuit_copies12 a h).2.1
  have c386 := (leafCircuit_copies12 a h).2.2.1
  have c387 := (leafCircuit_copies12 a h).2.2.2.1
  have c388 := (leafCircuit_copies12 a h).2.2.2.2.1
  have c389 := (leafCircuit_copies12 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_7 := arithEq_of_rows h (i := 7) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c387, ← c388, k3, ← c389] at e_8_7
  have e_12_5 := arithEq_of_rows h (i := 5) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c384, ← c385, ← c386] at e_12_5
  have hr := e_8_7
  simp only [e_12_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f99 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 13 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f100 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f101 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 2) = a (.virt 14) + a (.wire 13 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f102 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 2) = a (.wire 13 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f103 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 43) = bnot (a (.wire 13 2)) := by
  have c390 := (leafCircuit_copies12 a h).2.2.2.2.2.2.1
  have c391 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_10 := arithEq_of_rows h (i := 10) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_10
  simp only [← c390, k3, ← c391, k3, ← c392] at e_0_10
  have hr := e_0_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f104 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 11 59) = band (a (.wire 11 51)) (a (.wire 0 43)) := by
  have c393 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_14 := arithEq_of_rows h (i := 14) leafCircuit_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c393, ← c394, ← c395] at e_11_14
  have hr := e_11_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f105 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f106 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 1) = band (a (.virt 292)) (a (.wire 13 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f107 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 3) = band (a (.wire 13 1)) (a (.wire 11 59)) := by
  have c396 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_0 := arithEq_of_rows h (i := 0) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c396, ← c397, ← c398] at e_14_0
  have hr := e_14_0
  simp only [band]
  linear_combination hr

theorem leafCircuit_f108 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 35) = bor (a (.wire 8 31)) (a (.wire 14 3)) := by
  have c399 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_8 := arithEq_of_rows h (i := 8) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c402, ← c403, k3, ← c404] at e_8_8
  have e_12_6 := arithEq_of_rows h (i := 6) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c399, ← c400, ← c401] at e_12_6
  have hr := e_8_8
  simp only [e_12_6] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f109 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 13 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f110 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f111 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 1) = a (.virt 14) + a (.wire 13 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f112 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 13 1) = a (.wire 13 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f113 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 0 47) = bnot (a (.wire 13 1)) := by
  have c405 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_11 := arithEq_of_rows h (i := 11) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_11
  simp only [← c405, k3, ← c406, k3, ← c407] at e_0_11
  have hr := e_0_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f114 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 7) = band (a (.wire 11 59)) (a (.wire 0 47)) := by
  have c408 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_1 := arithEq_of_rows h (i := 1) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c408, ← c409, ← c410] at e_14_1
  have hr := e_14_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f115 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 207)) 2 := by
  have c411 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (leafCircuit_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (leafCircuit_copies13 a h).1
  have c417 := (leafCircuit_copies13 a h).2.1
  have c418 := (leafCircuit_copies13 a h).2.2.1
  have c419 := (leafCircuit_copies13 a h).2.2.2.1
  have c420 := (leafCircuit_copies13 a h).2.2.2.2.1
  have c421 := (leafCircuit_copies13 a h).2.2.2.2.2.1
  have c422 := (leafCircuit_copies13 a h).2.2.2.2.2.2.1
  have c423 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (leafCircuit_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (leafCircuit_copies14 a h).1
  have c449 := (leafCircuit_copies14 a h).2.1
  have c450 := (leafCircuit_copies14 a h).2.2.1
  have c451 := (leafCircuit_copies14 a h).2.2.2.1
  have c452 := (leafCircuit_copies14 a h).2.2.2.2.1
  have c453 := (leafCircuit_copies14 a h).2.2.2.2.2.1
  have c454 := (leafCircuit_copies14 a h).2.2.2.2.2.2.1
  have c455 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 15) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c411.trans k0
    · exact c412.trans k0
    · exact c413.trans k0
    · exact c414.trans k0
    · exact c415.trans k0
    · exact c416.trans k0
    · exact c417.trans k0
    · exact c418.trans k0
    · exact c419.trans k0
    · exact c420.trans k0
    · exact c421.trans k0
    · exact c422.trans k0
    · exact c423.trans k0
    · exact c424.trans k0
    · exact c425.trans k0
    · exact c426.trans k0
    · exact c427.trans k0
    · exact c428.trans k0
    · exact c429.trans k0
    · exact c430.trans k0
    · exact c431.trans k0
    · exact c432.trans k0
    · exact c433.trans k0
    · exact c434.trans k0
    · exact c435.trans k0
    · exact c436.trans k0
    · exact c437.trans k0
    · exact c438.trans k0
    · exact c439.trans k0
    · exact c440.trans k0
    · exact c441.trans k0
    · exact c442.trans k0
    · exact c443.trans k0
    · exact c444.trans k0
    · exact c445.trans k0
    · exact c446.trans k0
    · exact c447.trans k0
    · exact c448.trans k0
    · exact c449.trans k0
    · exact c450.trans k0
    · exact c451.trans k0
    · exact c452.trans k0
    · exact c453.trans k0
    · exact c454.trans k0
    · exact c455.trans k0
    · exact c456.trans k0
    · exact c457.trans k0
    · exact c458.trans k0
    · exact c459.trans k0
    · exact c460.trans k0
    · exact c461.trans k0
    · exact c462.trans k0
    · exact c463.trans k0
    · exact c464.trans k0
    · exact c465.trans k0
    · exact c466.trans k0
    · exact c467.trans k0
    )
  rwa [c468] at hr

theorem leafCircuit_f116 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 207)) (a (.virt 14)) (a (.virt 296)) (a (.virt 297)) := by
  have c469 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (leafCircuit_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (leafCircuit_copies15 a h).1
  have c481 := (leafCircuit_copies15 a h).2.1
  have c482 := (leafCircuit_copies15 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_12 := arithEq_of_rows h (i := 12) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c469, k3, ← c470, k3, ← c471] at e_0_12
  have e_0_13 := arithEq_of_rows h (i := 13) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c478, ← c479, k3, ← c480] at e_0_13
  have e_14_2 := arithEq_of_rows h (i := 2) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c472, ← c473, ← c474] at e_14_2
  have e_14_3 := arithEq_of_rows h (i := 3) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c475, ← c476, ← c477] at e_14_3
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_14_2
    linear_combination c481.trans k0 - hc
  · have hc := e_0_13
    simp only [e_0_12, e_14_3] at hc
    linear_combination c482.trans k0 - hc

theorem leafCircuit_f117 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 207)) (a (.virt 292)) (a (.virt 298)) (a (.virt 299)) := by
  have c483 := (leafCircuit_copies15 a h).2.2.2.1
  have c484 := (leafCircuit_copies15 a h).2.2.2.2.1
  have c485 := (leafCircuit_copies15 a h).2.2.2.2.2.1
  have c486 := (leafCircuit_copies15 a h).2.2.2.2.2.2.1
  have c487 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_0_14 := arithEq_of_rows h (i := 14) leafCircuit_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c483, k3, ← c484, k3, ← c485] at e_0_14
  have e_14_4 := arithEq_of_rows h (i := 4) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c489, ← c490, ← c491] at e_14_4
  have e_14_5 := arithEq_of_rows h (i := 5) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c492, ← c493, ← c494] at e_14_5
  have e_16_0 := arithEq_of_rows h (i := 0) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c486, ← c487, k3, ← c488, k3] at e_16_0
  have e_16_1 := arithEq_of_rows h (i := 1) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c495, ← c496, k3, ← c497] at e_16_1
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_14_4
    simp only [e_16_0] at hc
    linear_combination c498.trans k0 - hc
  · have hc := e_16_1
    simp only [e_0_14, e_14_5, e_16_0] at hc
    linear_combination c499.trans k0 - hc

theorem leafCircuit_f118 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 207)) (a (.virt 294)) (a (.virt 300)) (a (.virt 301)) := by
  have c500 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (leafCircuit_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (leafCircuit_copies16 a h).1
  have c513 := (leafCircuit_copies16 a h).2.1
  have c514 := (leafCircuit_copies16 a h).2.2.1
  have c515 := (leafCircuit_copies16 a h).2.2.2.1
  have c516 := (leafCircuit_copies16 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_14_6 := arithEq_of_rows h (i := 6) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c506, ← c507, ← c508] at e_14_6
  have e_14_7 := arithEq_of_rows h (i := 7) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c509, ← c510, ← c511] at e_14_7
  have e_16_2 := arithEq_of_rows h (i := 2) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c500, k3, ← c501, k3, ← c502] at e_16_2
  have e_16_3 := arithEq_of_rows h (i := 3) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c503, ← c504, k3, ← c505, k5] at e_16_3
  have e_16_4 := arithEq_of_rows h (i := 4) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c512, ← c513, k3, ← c514] at e_16_4
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_14_6
    simp only [e_16_3] at hc
    linear_combination c515.trans k0 - hc
  · have hc := e_16_4
    simp only [e_16_2, e_14_7, e_16_3] at hc
    linear_combination c516.trans k0 - hc

theorem leafCircuit_f119 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 207)) (a (.virt 295)) (a (.virt 302)) (a (.virt 303)) := by
  have c517 := (leafCircuit_copies16 a h).2.2.2.2.2.1
  have c518 := (leafCircuit_copies16 a h).2.2.2.2.2.2.1
  have c519 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c532 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_14_8 := arithEq_of_rows h (i := 8) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c523, ← c524, ← c525] at e_14_8
  have e_14_9 := arithEq_of_rows h (i := 9) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c526, ← c527, ← c528] at e_14_9
  have e_16_5 := arithEq_of_rows h (i := 5) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c517, k3, ← c518, k3, ← c519] at e_16_5
  have e_16_6 := arithEq_of_rows h (i := 6) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c520, ← c521, k3, ← c522, k6] at e_16_6
  have e_16_7 := arithEq_of_rows h (i := 7) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c529, ← c530, k3, ← c531] at e_16_7
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_14_8
    simp only [e_16_6] at hc
    linear_combination c532.trans k0 - hc
  · have hc := e_16_7
    simp only [e_16_5, e_14_9, e_16_6] at hc
    linear_combination c533.trans k0 - hc

theorem leafCircuit_f120 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 16 39) = bselect (a (.virt 296)) (a (.wire 9 12)) (a (.virt 15)) := by
  have c534 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c535 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_8 := arithEq_of_rows h (i := 8) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c534, ← c535, ← c536] at e_16_8
  have e_16_9 := arithEq_of_rows h (i := 9) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c537, ← c538, ← c539] at e_16_9
  have hr := e_16_9
  simp only [e_16_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f121 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 16 47) = bselect (a (.virt 296)) (a (.wire 9 13)) (a (.virt 16)) := by
  have c540 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c541 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (leafCircuit_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (leafCircuit_copies17 a h).1
  have c545 := (leafCircuit_copies17 a h).2.1
  have e_16_10 := arithEq_of_rows h (i := 10) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c540, ← c541, ← c542] at e_16_10
  have e_16_11 := arithEq_of_rows h (i := 11) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c543, ← c544, ← c545] at e_16_11
  have hr := e_16_11
  simp only [e_16_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f122 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 16 55) = bselect (a (.virt 296)) (a (.wire 9 14)) (a (.virt 17)) := by
  have c546 := (leafCircuit_copies17 a h).2.2.1
  have c547 := (leafCircuit_copies17 a h).2.2.2.1
  have c548 := (leafCircuit_copies17 a h).2.2.2.2.1
  have c549 := (leafCircuit_copies17 a h).2.2.2.2.2.1
  have c550 := (leafCircuit_copies17 a h).2.2.2.2.2.2.1
  have c551 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.1
  have e_16_12 := arithEq_of_rows h (i := 12) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c546, ← c547, ← c548] at e_16_12
  have e_16_13 := arithEq_of_rows h (i := 13) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c549, ← c550, ← c551] at e_16_13
  have hr := e_16_13
  simp only [e_16_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f123 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 3) = bselect (a (.virt 296)) (a (.wire 9 15)) (a (.virt 18)) := by
  have c552 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_14 := arithEq_of_rows h (i := 14) leafCircuit_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c552, ← c553, ← c554] at e_16_14
  have e_17_0 := arithEq_of_rows h (i := 0) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c555, ← c556, ← c557] at e_17_0
  have hr := e_17_0
  simp only [e_16_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f124 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 11) = bselect (a (.virt 296)) (a (.virt 15)) (a (.virt 19)) := by
  have c558 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_1 := arithEq_of_rows h (i := 1) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c558, ← c559, ← c560] at e_17_1
  have e_17_2 := arithEq_of_rows h (i := 2) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c561, ← c562, ← c563] at e_17_2
  have hr := e_17_2
  simp only [e_17_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f125 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 19) = bselect (a (.virt 298)) (a (.wire 9 12)) (a (.wire 17 11)) := by
  have c564 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_3 := arithEq_of_rows h (i := 3) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c564, ← c565, ← c566] at e_17_3
  have e_17_4 := arithEq_of_rows h (i := 4) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c567, ← c568, ← c569] at e_17_4
  have hr := e_17_4
  simp only [e_17_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f126 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 27) = bselect (a (.virt 296)) (a (.virt 16)) (a (.virt 20)) := by
  have c570 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (leafCircuit_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_17_5 := arithEq_of_rows h (i := 5) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c570, ← c571, ← c572] at e_17_5
  have e_17_6 := arithEq_of_rows h (i := 6) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c573, ← c574, ← c575] at e_17_6
  have hr := e_17_6
  simp only [e_17_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f127 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 35) = bselect (a (.virt 298)) (a (.wire 9 13)) (a (.wire 17 27)) := by
  have c576 := (leafCircuit_copies18 a h).1
  have c577 := (leafCircuit_copies18 a h).2.1
  have c578 := (leafCircuit_copies18 a h).2.2.1
  have c579 := (leafCircuit_copies18 a h).2.2.2.1
  have c580 := (leafCircuit_copies18 a h).2.2.2.2.1
  have c581 := (leafCircuit_copies18 a h).2.2.2.2.2.1
  have e_17_7 := arithEq_of_rows h (i := 7) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c576, ← c577, ← c578] at e_17_7
  have e_17_8 := arithEq_of_rows h (i := 8) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c579, ← c580, ← c581] at e_17_8
  have hr := e_17_8
  simp only [e_17_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f128 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 43) = bselect (a (.virt 296)) (a (.virt 17)) (a (.virt 21)) := by
  have c582 := (leafCircuit_copies18 a h).2.2.2.2.2.2.1
  have c583 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_9 := arithEq_of_rows h (i := 9) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c582, ← c583, ← c584] at e_17_9
  have e_17_10 := arithEq_of_rows h (i := 10) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c585, ← c586, ← c587] at e_17_10
  have hr := e_17_10
  simp only [e_17_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f129 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 51) = bselect (a (.virt 298)) (a (.wire 9 14)) (a (.wire 17 43)) := by
  have c588 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_11 := arithEq_of_rows h (i := 11) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c588, ← c589, ← c590] at e_17_11
  have e_17_12 := arithEq_of_rows h (i := 12) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c591, ← c592, ← c593] at e_17_12
  have hr := e_17_12
  simp only [e_17_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f130 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 17 59) = bselect (a (.virt 296)) (a (.virt 18)) (a (.virt 22)) := by
  have c594 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_13 := arithEq_of_rows h (i := 13) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c594, ← c595, ← c596] at e_17_13
  have e_17_14 := arithEq_of_rows h (i := 14) leafCircuit_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c597, ← c598, ← c599] at e_17_14
  have hr := e_17_14
  simp only [e_17_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f131 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 7) = bselect (a (.virt 298)) (a (.wire 9 15)) (a (.wire 17 59)) := by
  have c600 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_0 := arithEq_of_rows h (i := 0) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c600, ← c601, ← c602] at e_18_0
  have e_18_1 := arithEq_of_rows h (i := 1) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c603, ← c604, ← c605] at e_18_1
  have hr := e_18_1
  simp only [e_18_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f132 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) := by
  have c606 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (leafCircuit_copies19 a h).1
  have c609 := (leafCircuit_copies19 a h).2.1
  have c610 := (leafCircuit_copies19 a h).2.2.1
  have c611 := (leafCircuit_copies19 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_9 := arithEq_of_rows h (i := 9) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c609, ← c610, k3, ← c611] at e_8_9
  have e_12_7 := arithEq_of_rows h (i := 7) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c606, ← c607, ← c608] at e_12_7
  have hr := e_8_9
  simp only [e_12_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f133 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 15) = bselect (a (.wire 8 39)) (a (.virt 19)) (a (.virt 23)) := by
  have c612 := (leafCircuit_copies19 a h).2.2.2.2.1
  have c613 := (leafCircuit_copies19 a h).2.2.2.2.2.1
  have c614 := (leafCircuit_copies19 a h).2.2.2.2.2.2.1
  have c615 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have e_18_2 := arithEq_of_rows h (i := 2) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c612, ← c613, ← c614] at e_18_2
  have e_18_3 := arithEq_of_rows h (i := 3) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c615, ← c616, ← c617] at e_18_3
  have hr := e_18_3
  simp only [e_18_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f134 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 23) = bselect (a (.virt 300)) (a (.wire 9 12)) (a (.wire 18 15)) := by
  have c618 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_4 := arithEq_of_rows h (i := 4) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c618, ← c619, ← c620] at e_18_4
  have e_18_5 := arithEq_of_rows h (i := 5) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c621, ← c622, ← c623] at e_18_5
  have hr := e_18_5
  simp only [e_18_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f135 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) := by
  have c606 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (leafCircuit_copies19 a h).1
  have c609 := (leafCircuit_copies19 a h).2.1
  have c610 := (leafCircuit_copies19 a h).2.2.1
  have c611 := (leafCircuit_copies19 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_9 := arithEq_of_rows h (i := 9) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c609, ← c610, k3, ← c611] at e_8_9
  have e_12_7 := arithEq_of_rows h (i := 7) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c606, ← c607, ← c608] at e_12_7
  have hr := e_8_9
  simp only [e_12_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f136 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 31) = bselect (a (.wire 8 39)) (a (.virt 20)) (a (.virt 24)) := by
  have c624 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_6 := arithEq_of_rows h (i := 6) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c624, ← c625, ← c626] at e_18_6
  have e_18_7 := arithEq_of_rows h (i := 7) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c627, ← c628, ← c629] at e_18_7
  have hr := e_18_7
  simp only [e_18_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f137 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 39) = bselect (a (.virt 300)) (a (.wire 9 13)) (a (.wire 18 31)) := by
  have c630 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_8 := arithEq_of_rows h (i := 8) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c630, ← c631, ← c632] at e_18_8
  have e_18_9 := arithEq_of_rows h (i := 9) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c633, ← c634, ← c635] at e_18_9
  have hr := e_18_9
  simp only [e_18_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f138 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) := by
  have c606 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (leafCircuit_copies19 a h).1
  have c609 := (leafCircuit_copies19 a h).2.1
  have c610 := (leafCircuit_copies19 a h).2.2.1
  have c611 := (leafCircuit_copies19 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_9 := arithEq_of_rows h (i := 9) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c609, ← c610, k3, ← c611] at e_8_9
  have e_12_7 := arithEq_of_rows h (i := 7) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c606, ← c607, ← c608] at e_12_7
  have hr := e_8_9
  simp only [e_12_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f139 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 47) = bselect (a (.wire 8 39)) (a (.virt 21)) (a (.virt 25)) := by
  have c636 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (leafCircuit_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (leafCircuit_copies20 a h).1
  have c641 := (leafCircuit_copies20 a h).2.1
  have e_18_10 := arithEq_of_rows h (i := 10) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c636, ← c637, ← c638] at e_18_10
  have e_18_11 := arithEq_of_rows h (i := 11) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c639, ← c640, ← c641] at e_18_11
  have hr := e_18_11
  simp only [e_18_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f140 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 18 55) = bselect (a (.virt 300)) (a (.wire 9 14)) (a (.wire 18 47)) := by
  have c642 := (leafCircuit_copies20 a h).2.2.1
  have c643 := (leafCircuit_copies20 a h).2.2.2.1
  have c644 := (leafCircuit_copies20 a h).2.2.2.2.1
  have c645 := (leafCircuit_copies20 a h).2.2.2.2.2.1
  have c646 := (leafCircuit_copies20 a h).2.2.2.2.2.2.1
  have c647 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.1
  have e_18_12 := arithEq_of_rows h (i := 12) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c642, ← c643, ← c644] at e_18_12
  have e_18_13 := arithEq_of_rows h (i := 13) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c645, ← c646, ← c647] at e_18_13
  have hr := e_18_13
  simp only [e_18_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f141 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) := by
  have c606 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (leafCircuit_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (leafCircuit_copies19 a h).1
  have c609 := (leafCircuit_copies19 a h).2.1
  have c610 := (leafCircuit_copies19 a h).2.2.1
  have c611 := (leafCircuit_copies19 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_9 := arithEq_of_rows h (i := 9) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c609, ← c610, k3, ← c611] at e_8_9
  have e_12_7 := arithEq_of_rows h (i := 7) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c606, ← c607, ← c608] at e_12_7
  have hr := e_8_9
  simp only [e_12_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f142 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 3) = bselect (a (.wire 8 39)) (a (.virt 22)) (a (.virt 26)) := by
  have c648 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_18_14 := arithEq_of_rows h (i := 14) leafCircuit_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c648, ← c649, ← c650] at e_18_14
  have e_19_0 := arithEq_of_rows h (i := 0) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c651, ← c652, ← c653] at e_19_0
  have hr := e_19_0
  simp only [e_18_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f143 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 11) = bselect (a (.virt 300)) (a (.wire 9 15)) (a (.wire 19 3)) := by
  have c654 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_1 := arithEq_of_rows h (i := 1) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c654, ← c655, ← c656] at e_19_1
  have e_19_2 := arithEq_of_rows h (i := 2) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c657, ← c658, ← c659] at e_19_2
  have hr := e_19_2
  simp only [e_19_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f144 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 19) = bselect (a (.virt 302)) (a (.wire 9 12)) (a (.virt 23)) := by
  have c660 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_3 := arithEq_of_rows h (i := 3) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c660, ← c661, ← c662] at e_19_3
  have e_19_4 := arithEq_of_rows h (i := 4) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c663, ← c664, ← c665] at e_19_4
  have hr := e_19_4
  simp only [e_19_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f145 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 27) = bselect (a (.virt 302)) (a (.wire 9 13)) (a (.virt 24)) := by
  have c666 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (leafCircuit_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_19_5 := arithEq_of_rows h (i := 5) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c666, ← c667, ← c668] at e_19_5
  have e_19_6 := arithEq_of_rows h (i := 6) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c669, ← c670, ← c671] at e_19_6
  have hr := e_19_6
  simp only [e_19_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f146 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 35) = bselect (a (.virt 302)) (a (.wire 9 14)) (a (.virt 25)) := by
  have c672 := (leafCircuit_copies21 a h).1
  have c673 := (leafCircuit_copies21 a h).2.1
  have c674 := (leafCircuit_copies21 a h).2.2.1
  have c675 := (leafCircuit_copies21 a h).2.2.2.1
  have c676 := (leafCircuit_copies21 a h).2.2.2.2.1
  have c677 := (leafCircuit_copies21 a h).2.2.2.2.2.1
  have e_19_7 := arithEq_of_rows h (i := 7) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c672, ← c673, ← c674] at e_19_7
  have e_19_8 := arithEq_of_rows h (i := 8) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c675, ← c676, ← c677] at e_19_8
  have hr := e_19_8
  simp only [e_19_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f147 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 43) = bselect (a (.virt 302)) (a (.wire 9 15)) (a (.virt 26)) := by
  have c678 := (leafCircuit_copies21 a h).2.2.2.2.2.2.1
  have c679 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_9 := arithEq_of_rows h (i := 9) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c678, ← c679, ← c680] at e_19_9
  have e_19_10 := arithEq_of_rows h (i := 10) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c681, ← c682, ← c683] at e_19_10
  have hr := e_19_10
  simp only [e_19_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f148 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 23 12) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 0 ∧ a (.wire 23 13) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 1 ∧ a (.wire 23 14) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 2 ∧ a (.wire 23 15) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 3) := by
  have c684 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (leafCircuit_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (leafCircuit_copies22 a h).1
  have c705 := (leafCircuit_copies22 a h).2.1
  have c706 := (leafCircuit_copies22 a h).2.2.1
  have c707 := (leafCircuit_copies22 a h).2.2.2.1
  have c708 := (leafCircuit_copies22 a h).2.2.2.2.1
  have c709 := (leafCircuit_copies22 a h).2.2.2.2.2.1
  have c710 := (leafCircuit_copies22 a h).2.2.2.2.2.2.1
  have c711 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c733 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (leafCircuit_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c736 := (leafCircuit_copies23 a h).1
  have c737 := (leafCircuit_copies23 a h).2.1
  have c738 := (leafCircuit_copies23 a h).2.2.1
  have c739 := (leafCircuit_copies23 a h).2.2.2.1
  have c740 := (leafCircuit_copies23 a h).2.2.2.2.1
  have c741 := (leafCircuit_copies23 a h).2.2.2.2.2.1
  have c742 := (leafCircuit_copies23 a h).2.2.2.2.2.2.1
  have c743 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.1
  have c745 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_8_10 := arithEq_of_rows h (i := 10) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c696, ← c697, k3, ← c698] at e_8_10
  have e_8_11 := arithEq_of_rows h (i := 11) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c699, ← c700, k3, ← c701] at e_8_11
  have e_8_12 := arithEq_of_rows h (i := 12) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c702, ← c703, k3, ← c704] at e_8_12
  have e_8_13 := arithEq_of_rows h (i := 13) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c705, ← c706, k3, ← c707] at e_8_13
  have e_8_14 := arithEq_of_rows h (i := 14) leafCircuit_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c708, ← c709, k3, ← c710] at e_8_14
  have e_21_0 := arithEq_of_rows h (i := 0) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c711, ← c712, k3, ← c713] at e_21_0
  have e_21_1 := arithEq_of_rows h (i := 1) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c714, ← c715, k3, ← c716] at e_21_1
  have e_21_2 := arithEq_of_rows h (i := 2) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c717, ← c718, k3, ← c719] at e_21_2
  have e_21_3 := arithEq_of_rows h (i := 3) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c732, ← c733, k3, ← c734, k3] at e_21_3
  have hin0 : poseidon2In a 20 = addBlock (fun _ => 0) [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7)] :=
    poseidon2In_first c684.symm c685.symm c686.symm c687.symm c688.symm c689.symm c690.symm c691.symm (c692.symm.trans k0) (c693.symm.trans k0) (c694.symm.trans k0) (c695.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 20) rfl rfl hin0
  have hin1 : poseidon2In a 22 = addBlock (poseidon2Out a 20) [a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] :=
    poseidon2In_chain (by linear_combination -c720 + e_8_10) (by linear_combination -c721 + e_8_11) (by linear_combination -c722 + e_8_12) (by linear_combination -c723 + e_8_13) (by linear_combination -c724 + e_8_14) (by linear_combination -c725 + e_21_0) (by linear_combination -c726 + e_21_1) (by linear_combination -c727 + e_21_2) c728.symm c729.symm c730.symm c731.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 22) rfl rfl hin1
  have hin2 : poseidon2In a 23 = addBlock (poseidon2Out a 22) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c735 + e_21_3) (c736.symm.trans (add_zero _).symm) (c737.symm.trans (add_zero _).symm) (c738.symm.trans (add_zero _).symm) (c739.symm.trans (add_zero _).symm) (c740.symm.trans (add_zero _).symm) (c741.symm.trans (add_zero _).symm) (c742.symm.trans (add_zero _).symm) c743.symm c744.symm c745.symm c746.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 23) rfl rfl hin2
  have hpad : pad10 [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] =
      [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7)] ++ ([a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f149 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 51) = bselect (a (.wire 8 35)) (a (.wire 23 12)) (a (.wire 9 12)) := by
  have c747 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c751 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_11 := arithEq_of_rows h (i := 11) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c747, ← c748, ← c749] at e_19_11
  have e_19_12 := arithEq_of_rows h (i := 12) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c750, ← c751, ← c752] at e_19_12
  have hr := e_19_12
  simp only [e_19_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f150 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 19 59) = bselect (a (.wire 8 35)) (a (.wire 23 13)) (a (.wire 9 13)) := by
  have c753 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_13 := arithEq_of_rows h (i := 13) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c753, ← c754, ← c755] at e_19_13
  have e_19_14 := arithEq_of_rows h (i := 14) leafCircuit_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c756, ← c757, ← c758] at e_19_14
  have hr := e_19_14
  simp only [e_19_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f151 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 7) = bselect (a (.wire 8 35)) (a (.wire 23 14)) (a (.wire 9 14)) := by
  have c759 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_24_0 := arithEq_of_rows h (i := 0) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_0
  simp only [← c759, ← c760, ← c761] at e_24_0
  have e_24_1 := arithEq_of_rows h (i := 1) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_1
  simp only [← c762, ← c763, ← c764] at e_24_1
  have hr := e_24_1
  simp only [e_24_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f152 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 15) = bselect (a (.wire 8 35)) (a (.wire 23 15)) (a (.wire 9 15)) := by
  have c765 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (leafCircuit_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (leafCircuit_copies24 a h).1
  have c769 := (leafCircuit_copies24 a h).2.1
  have c770 := (leafCircuit_copies24 a h).2.2.1
  have e_24_2 := arithEq_of_rows h (i := 2) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c765, ← c766, ← c767] at e_24_2
  have e_24_3 := arithEq_of_rows h (i := 3) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_3
  simp only [← c768, ← c769, ← c770] at e_24_3
  have hr := e_24_3
  simp only [e_24_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f153 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 25 1), a (.wire 25 2), a (.wire 25 3), a (.wire 25 4), a (.wire 25 5)] := by
  have c771 := (leafCircuit_copies24 a h).2.2.2.1
  have c772 := (leafCircuit_copies24 a h).2.2.2.2.1
  have c773 := (leafCircuit_copies24 a h).2.2.2.2.2.1
  have c774 := (leafCircuit_copies24 a h).2.2.2.2.2.2.1
  have c775 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have c778 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c781 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (leafCircuit_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (leafCircuit_copies25 a h).1
  have c801 := (leafCircuit_copies25 a h).2.1
  have c802 := (leafCircuit_copies25 a h).2.2.1
  have c803 := (leafCircuit_copies25 a h).2.2.2.1
  have c804 := (leafCircuit_copies25 a h).2.2.2.2.1
  have c805 := (leafCircuit_copies25 a h).2.2.2.2.2.1
  have c806 := (leafCircuit_copies25 a h).2.2.2.2.2.2.1
  have c807 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c814 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 25) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c771.trans k0
    · exact c772.trans k0
    · exact c773.trans k0
    · exact c774.trans k0
    · exact c775.trans k0
    · exact c776.trans k0
    · exact c777.trans k0
    · exact c778.trans k0
    · exact c779.trans k0
    · exact c780.trans k0
    · exact c781.trans k0
    · exact c782.trans k0
    · exact c783.trans k0
    · exact c784.trans k0
    · exact c785.trans k0
    · exact c786.trans k0
    · exact c787.trans k0
    · exact c788.trans k0
    · exact c789.trans k0
    · exact c790.trans k0
    · exact c791.trans k0
    · exact c792.trans k0
    · exact c793.trans k0
    · exact c794.trans k0
    · exact c795.trans k0
    · exact c796.trans k0
    · exact c797.trans k0
    · exact c798.trans k0
    · exact c799.trans k0
    · exact c800.trans k0
    · exact c801.trans k0
    · exact c802.trans k0
    · exact c803.trans k0
    · exact c804.trans k0
    · exact c805.trans k0
    · exact c806.trans k0
    · exact c807.trans k0
    · exact c808.trans k0
    · exact c809.trans k0
    · exact c810.trans k0
    · exact c811.trans k0
    · exact c812.trans k0
    · exact c813.trans k0
    · exact c814.trans k0
    · exact c815.trans k0
    · exact c816.trans k0
    · exact c817.trans k0
    · exact c818.trans k0
    · exact c819.trans k0
    · exact c820.trans k0
    · exact c821.trans k0
    · exact c822.trans k0
    · exact c823.trans k0
    · exact c824.trans k0
    )
  rwa [c825] at hr

theorem leafCircuit_f154 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f155 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 5) = band (a (.virt 292)) (a (.wire 25 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f156 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 5) = band (a (.wire 25 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f157 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 5) = bor (a (.virt 14)) (a (.wire 25 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f158 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 25 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f159 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f160 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 5) = a (.virt 14) + a (.wire 25 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f161 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 5) = a (.wire 25 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f162 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 19) = bnot (a (.wire 25 5)) := by
  have c826 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_4 := arithEq_of_rows h (i := 4) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_4
  simp only [← c826, k3, ← c827, k3, ← c828] at e_24_4
  have hr := e_24_4
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f163 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 19) = band (a (.virt 292)) (a (.wire 24 19)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f164 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f165 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 4) = band (a (.virt 292)) (a (.wire 25 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f166 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 43) = band (a (.wire 25 4)) (a (.wire 24 19)) := by
  have c829 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (leafCircuit_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_14_10 := arithEq_of_rows h (i := 10) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c829, ← c830, ← c831] at e_14_10
  have hr := e_14_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f167 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 19) = bor (a (.wire 25 5)) (a (.wire 14 43)) := by
  have c832 := (leafCircuit_copies26 a h).1
  have c833 := (leafCircuit_copies26 a h).2.1
  have c834 := (leafCircuit_copies26 a h).2.2.1
  have c835 := (leafCircuit_copies26 a h).2.2.2.1
  have c836 := (leafCircuit_copies26 a h).2.2.2.2.1
  have c837 := (leafCircuit_copies26 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_8 := arithEq_of_rows h (i := 8) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c832, ← c833, ← c834] at e_12_8
  have e_21_4 := arithEq_of_rows h (i := 4) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c835, ← c836, k3, ← c837] at e_21_4
  have hr := e_21_4
  simp only [e_12_8] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f168 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 25 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f169 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f170 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 4) = a (.virt 14) + a (.wire 25 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f171 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 4) = a (.wire 25 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f172 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 23) = bnot (a (.wire 25 4)) := by
  have c838 := (leafCircuit_copies26 a h).2.2.2.2.2.2.1
  have c839 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_5 := arithEq_of_rows h (i := 5) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c838, k3, ← c839, k3, ← c840] at e_24_5
  have hr := e_24_5
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f173 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 47) = band (a (.wire 24 19)) (a (.wire 24 23)) := by
  have c841 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have c842 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_11 := arithEq_of_rows h (i := 11) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c841, ← c842, ← c843] at e_14_11
  have hr := e_14_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f174 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f175 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 3) = band (a (.virt 292)) (a (.wire 25 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f176 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 51) = band (a (.wire 25 3)) (a (.wire 14 47)) := by
  have c844 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c845 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_12 := arithEq_of_rows h (i := 12) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c844, ← c845, ← c846] at e_14_12
  have hr := e_14_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f177 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 23) = bor (a (.wire 21 19)) (a (.wire 14 51)) := by
  have c847 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_9 := arithEq_of_rows h (i := 9) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c847, ← c848, ← c849] at e_12_9
  have e_21_5 := arithEq_of_rows h (i := 5) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c850, ← c851, k3, ← c852] at e_21_5
  have hr := e_21_5
  simp only [e_12_9] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f178 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 25 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f179 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f180 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 3) = a (.virt 14) + a (.wire 25 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f181 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 3) = a (.wire 25 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f182 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 27) = bnot (a (.wire 25 3)) := by
  have c853 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_6 := arithEq_of_rows h (i := 6) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_6
  simp only [← c853, k3, ← c854, k3, ← c855] at e_24_6
  have hr := e_24_6
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f183 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 55) = band (a (.wire 14 47)) (a (.wire 24 27)) := by
  have c856 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_13 := arithEq_of_rows h (i := 13) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c856, ← c857, ← c858] at e_14_13
  have hr := e_14_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f184 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f185 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 2) = band (a (.virt 292)) (a (.wire 25 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f186 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 14 59) = band (a (.wire 25 2)) (a (.wire 14 55)) := by
  have c859 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c860 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_14 := arithEq_of_rows h (i := 14) leafCircuit_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c859, ← c860, ← c861] at e_14_14
  have hr := e_14_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f187 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 27) = bor (a (.wire 21 23)) (a (.wire 14 59)) := by
  have c862 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c863 := (leafCircuit_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (leafCircuit_copies27 a h).1
  have c865 := (leafCircuit_copies27 a h).2.1
  have c866 := (leafCircuit_copies27 a h).2.2.1
  have c867 := (leafCircuit_copies27 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_10 := arithEq_of_rows h (i := 10) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c862, ← c863, ← c864] at e_12_10
  have e_21_6 := arithEq_of_rows h (i := 6) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c865, ← c866, k3, ← c867] at e_21_6
  have hr := e_21_6
  simp only [e_12_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f188 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 25 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f189 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f190 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 2) = a (.virt 14) + a (.wire 25 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f191 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 2) = a (.wire 25 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f192 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 31) = bnot (a (.wire 25 2)) := by
  have c868 := (leafCircuit_copies27 a h).2.2.2.2.1
  have c869 := (leafCircuit_copies27 a h).2.2.2.2.2.1
  have c870 := (leafCircuit_copies27 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_7 := arithEq_of_rows h (i := 7) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_7
  simp only [← c868, k3, ← c869, k3, ← c870] at e_24_7
  have hr := e_24_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f193 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 3) = band (a (.wire 14 55)) (a (.wire 24 31)) := by
  have c871 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.1
  have c873 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have e_26_0 := arithEq_of_rows h (i := 0) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c871, ← c872, ← c873] at e_26_0
  have hr := e_26_0
  simp only [band]
  linear_combination hr

theorem leafCircuit_f194 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f195 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 25 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f196 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 26 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f197 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 27) = bor (a (.wire 21 27)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f198 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 25 1) = a (.virt 292) * a (.wire 25 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f199 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 7) = a (.virt 294) * a (.wire 25 1) := by
  have c874 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c876 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_26_1 := arithEq_of_rows h (i := 1) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c874, k5, ← c875, ← c876, k5] at e_26_1
  have hr := e_26_1
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f200 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 31) = a (.virt 292) + a (.wire 25 1) := by
  have c877 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_21_7 := arithEq_of_rows h (i := 7) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c877, k3, ← c878, k3, ← c879] at e_21_7
  have hr := e_21_7
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f201 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 35) = a (.wire 21 31) - a (.wire 26 7) := by
  have c880 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_8 := arithEq_of_rows h (i := 8) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_8
  simp only [← c880, ← c881, k3, ← c882] at e_24_8
  have hr := e_24_8
  linear_combination hr

theorem leafCircuit_f202 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 24 39) = bnot (a (.wire 24 35)) := by
  have c883 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_9 := arithEq_of_rows h (i := 9) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_9
  simp only [← c883, k3, ← c884, k3, ← c885] at e_24_9
  have hr := e_24_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f203 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 11) = band (a (.wire 26 3)) (a (.wire 24 39)) := by
  have c886 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_2 := arithEq_of_rows h (i := 2) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c886, ← c887, ← c888] at e_26_2
  have hr := e_26_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f204 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 208)) 2 := by
  have c889 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (leafCircuit_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (leafCircuit_copies28 a h).1
  have c897 := (leafCircuit_copies28 a h).2.1
  have c898 := (leafCircuit_copies28 a h).2.2.1
  have c899 := (leafCircuit_copies28 a h).2.2.2.1
  have c900 := (leafCircuit_copies28 a h).2.2.2.2.1
  have c901 := (leafCircuit_copies28 a h).2.2.2.2.2.1
  have c902 := (leafCircuit_copies28 a h).2.2.2.2.2.2.1
  have c903 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.1
  have c904 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have c907 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c925 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c927 := (leafCircuit_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c928 := (leafCircuit_copies29 a h).1
  have c929 := (leafCircuit_copies29 a h).2.1
  have c930 := (leafCircuit_copies29 a h).2.2.1
  have c931 := (leafCircuit_copies29 a h).2.2.2.1
  have c932 := (leafCircuit_copies29 a h).2.2.2.2.1
  have c933 := (leafCircuit_copies29 a h).2.2.2.2.2.1
  have c934 := (leafCircuit_copies29 a h).2.2.2.2.2.2.1
  have c935 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.1
  have c937 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 27) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c889.trans k0
    · exact c890.trans k0
    · exact c891.trans k0
    · exact c892.trans k0
    · exact c893.trans k0
    · exact c894.trans k0
    · exact c895.trans k0
    · exact c896.trans k0
    · exact c897.trans k0
    · exact c898.trans k0
    · exact c899.trans k0
    · exact c900.trans k0
    · exact c901.trans k0
    · exact c902.trans k0
    · exact c903.trans k0
    · exact c904.trans k0
    · exact c905.trans k0
    · exact c906.trans k0
    · exact c907.trans k0
    · exact c908.trans k0
    · exact c909.trans k0
    · exact c910.trans k0
    · exact c911.trans k0
    · exact c912.trans k0
    · exact c913.trans k0
    · exact c914.trans k0
    · exact c915.trans k0
    · exact c916.trans k0
    · exact c917.trans k0
    · exact c918.trans k0
    · exact c919.trans k0
    · exact c920.trans k0
    · exact c921.trans k0
    · exact c922.trans k0
    · exact c923.trans k0
    · exact c924.trans k0
    · exact c925.trans k0
    · exact c926.trans k0
    · exact c927.trans k0
    · exact c928.trans k0
    · exact c929.trans k0
    · exact c930.trans k0
    · exact c931.trans k0
    · exact c932.trans k0
    · exact c933.trans k0
    · exact c934.trans k0
    · exact c935.trans k0
    · exact c936.trans k0
    · exact c937.trans k0
    · exact c938.trans k0
    · exact c939.trans k0
    · exact c940.trans k0
    · exact c941.trans k0
    · exact c942.trans k0
    · exact c943.trans k0
    · exact c944.trans k0
    · exact c945.trans k0
    )
  rwa [c946] at hr

theorem leafCircuit_f205 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 208)) (a (.virt 14)) (a (.virt 304)) (a (.virt 305)) := by
  have c947 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c953 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c956 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c959 := (leafCircuit_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (leafCircuit_copies30 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_10 := arithEq_of_rows h (i := 10) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_10
  simp only [← c947, k3, ← c948, k3, ← c949] at e_24_10
  have e_24_11 := arithEq_of_rows h (i := 11) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_11
  simp only [← c956, ← c957, k3, ← c958] at e_24_11
  have e_26_3 := arithEq_of_rows h (i := 3) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c950, ← c951, ← c952] at e_26_3
  have e_26_4 := arithEq_of_rows h (i := 4) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c953, ← c954, ← c955] at e_26_4
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_26_3
    linear_combination c959.trans k0 - hc
  · have hc := e_24_11
    simp only [e_24_10, e_26_4] at hc
    linear_combination c960.trans k0 - hc

theorem leafCircuit_f206 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 208)) (a (.virt 292)) (a (.virt 306)) (a (.virt 307)) := by
  have c961 := (leafCircuit_copies30 a h).2.1
  have c962 := (leafCircuit_copies30 a h).2.2.1
  have c963 := (leafCircuit_copies30 a h).2.2.2.1
  have c964 := (leafCircuit_copies30 a h).2.2.2.2.1
  have c965 := (leafCircuit_copies30 a h).2.2.2.2.2.1
  have c966 := (leafCircuit_copies30 a h).2.2.2.2.2.2.1
  have c967 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  have c971 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c977 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_24_12 := arithEq_of_rows h (i := 12) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_12
  simp only [← c961, k3, ← c962, k3, ← c963] at e_24_12
  have e_24_13 := arithEq_of_rows h (i := 13) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_13
  simp only [← c964, ← c965, k3, ← c966, k3] at e_24_13
  have e_24_14 := arithEq_of_rows h (i := 14) leafCircuit_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_14
  simp only [← c973, ← c974, k3, ← c975] at e_24_14
  have e_26_5 := arithEq_of_rows h (i := 5) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c967, ← c968, ← c969] at e_26_5
  have e_26_6 := arithEq_of_rows h (i := 6) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c970, ← c971, ← c972] at e_26_6
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_26_5
    simp only [e_24_13] at hc
    linear_combination c976.trans k0 - hc
  · have hc := e_24_14
    simp only [e_24_12, e_26_6, e_24_13] at hc
    linear_combination c977.trans k0 - hc

theorem leafCircuit_f207 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 208)) (a (.virt 294)) (a (.virt 308)) (a (.virt 309)) := by
  have c978 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c987 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c989 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c990 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (leafCircuit_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (leafCircuit_copies31 a h).1
  have c993 := (leafCircuit_copies31 a h).2.1
  have c994 := (leafCircuit_copies31 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_26_7 := arithEq_of_rows h (i := 7) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c984, ← c985, ← c986] at e_26_7
  have e_26_8 := arithEq_of_rows h (i := 8) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c987, ← c988, ← c989] at e_26_8
  have e_28_0 := arithEq_of_rows h (i := 0) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c978, k3, ← c979, k3, ← c980] at e_28_0
  have e_28_1 := arithEq_of_rows h (i := 1) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c981, ← c982, k3, ← c983, k5] at e_28_1
  have e_28_2 := arithEq_of_rows h (i := 2) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c990, ← c991, k3, ← c992] at e_28_2
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_26_7
    simp only [e_28_1] at hc
    linear_combination c993.trans k0 - hc
  · have hc := e_28_2
    simp only [e_28_0, e_26_8, e_28_1] at hc
    linear_combination c994.trans k0 - hc

theorem leafCircuit_f208 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 208)) (a (.virt 295)) (a (.virt 310)) (a (.virt 311)) := by
  have c995 := (leafCircuit_copies31 a h).2.2.2.1
  have c996 := (leafCircuit_copies31 a h).2.2.2.2.1
  have c997 := (leafCircuit_copies31 a h).2.2.2.2.2.1
  have c998 := (leafCircuit_copies31 a h).2.2.2.2.2.2.1
  have c999 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have c1002 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1005 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1008 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1011 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_26_9 := arithEq_of_rows h (i := 9) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1001, ← c1002, ← c1003] at e_26_9
  have e_26_10 := arithEq_of_rows h (i := 10) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1004, ← c1005, ← c1006] at e_26_10
  have e_28_3 := arithEq_of_rows h (i := 3) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c995, k3, ← c996, k3, ← c997] at e_28_3
  have e_28_4 := arithEq_of_rows h (i := 4) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c998, ← c999, k3, ← c1000, k6] at e_28_4
  have e_28_5 := arithEq_of_rows h (i := 5) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1007, ← c1008, k3, ← c1009] at e_28_5
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_26_9
    simp only [e_28_4] at hc
    linear_combination c1010.trans k0 - hc
  · have hc := e_28_5
    simp only [e_28_3, e_26_10, e_28_4] at hc
    linear_combination c1011.trans k0 - hc

theorem leafCircuit_f209 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 28 31) = bselect (a (.virt 304)) (a (.wire 19 51)) (a (.virt 27)) := by
  have c1012 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1017 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_6 := arithEq_of_rows h (i := 6) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1012, ← c1013, ← c1014] at e_28_6
  have e_28_7 := arithEq_of_rows h (i := 7) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1015, ← c1016, ← c1017] at e_28_7
  have hr := e_28_7
  simp only [e_28_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f210 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 28 39) = bselect (a (.virt 304)) (a (.wire 19 59)) (a (.virt 28)) := by
  have c1018 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1019 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1022 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1023 := (leafCircuit_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_28_8 := arithEq_of_rows h (i := 8) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1018, ← c1019, ← c1020] at e_28_8
  have e_28_9 := arithEq_of_rows h (i := 9) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1021, ← c1022, ← c1023] at e_28_9
  have hr := e_28_9
  simp only [e_28_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f211 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 28 47) = bselect (a (.virt 304)) (a (.wire 24 7)) (a (.virt 29)) := by
  have c1024 := (leafCircuit_copies32 a h).1
  have c1025 := (leafCircuit_copies32 a h).2.1
  have c1026 := (leafCircuit_copies32 a h).2.2.1
  have c1027 := (leafCircuit_copies32 a h).2.2.2.1
  have c1028 := (leafCircuit_copies32 a h).2.2.2.2.1
  have c1029 := (leafCircuit_copies32 a h).2.2.2.2.2.1
  have e_28_10 := arithEq_of_rows h (i := 10) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1024, ← c1025, ← c1026] at e_28_10
  have e_28_11 := arithEq_of_rows h (i := 11) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1027, ← c1028, ← c1029] at e_28_11
  have hr := e_28_11
  simp only [e_28_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f212 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 28 55) = bselect (a (.virt 304)) (a (.wire 24 15)) (a (.virt 30)) := by
  have c1030 := (leafCircuit_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_12 := arithEq_of_rows h (i := 12) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1030, ← c1031, ← c1032] at e_28_12
  have e_28_13 := arithEq_of_rows h (i := 13) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1033, ← c1034, ← c1035] at e_28_13
  have hr := e_28_13
  simp only [e_28_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f213 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 3) = bselect (a (.virt 304)) (a (.virt 27)) (a (.virt 31)) := by
  have c1036 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1041 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_14 := arithEq_of_rows h (i := 14) leafCircuit_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1036, ← c1037, ← c1038] at e_28_14
  have e_29_0 := arithEq_of_rows h (i := 0) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1039, ← c1040, ← c1041] at e_29_0
  have hr := e_29_0
  simp only [e_28_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f214 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 11) = bselect (a (.virt 306)) (a (.wire 19 51)) (a (.wire 29 3)) := by
  have c1042 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_1 := arithEq_of_rows h (i := 1) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1042, ← c1043, ← c1044] at e_29_1
  have e_29_2 := arithEq_of_rows h (i := 2) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1045, ← c1046, ← c1047] at e_29_2
  have hr := e_29_2
  simp only [e_29_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f215 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 19) = bselect (a (.virt 304)) (a (.virt 28)) (a (.virt 32)) := by
  have c1048 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_3 := arithEq_of_rows h (i := 3) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1048, ← c1049, ← c1050] at e_29_3
  have e_29_4 := arithEq_of_rows h (i := 4) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1051, ← c1052, ← c1053] at e_29_4
  have hr := e_29_4
  simp only [e_29_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f216 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 27) = bselect (a (.virt 306)) (a (.wire 19 59)) (a (.wire 29 19)) := by
  have c1054 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (leafCircuit_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (leafCircuit_copies33 a h).1
  have c1057 := (leafCircuit_copies33 a h).2.1
  have c1058 := (leafCircuit_copies33 a h).2.2.1
  have c1059 := (leafCircuit_copies33 a h).2.2.2.1
  have e_29_5 := arithEq_of_rows h (i := 5) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1054, ← c1055, ← c1056] at e_29_5
  have e_29_6 := arithEq_of_rows h (i := 6) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1057, ← c1058, ← c1059] at e_29_6
  have hr := e_29_6
  simp only [e_29_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f217 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 35) = bselect (a (.virt 304)) (a (.virt 29)) (a (.virt 33)) := by
  have c1060 := (leafCircuit_copies33 a h).2.2.2.2.1
  have c1061 := (leafCircuit_copies33 a h).2.2.2.2.2.1
  have c1062 := (leafCircuit_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have e_29_7 := arithEq_of_rows h (i := 7) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c1060, ← c1061, ← c1062] at e_29_7
  have e_29_8 := arithEq_of_rows h (i := 8) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c1063, ← c1064, ← c1065] at e_29_8
  have hr := e_29_8
  simp only [e_29_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f218 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 43) = bselect (a (.virt 306)) (a (.wire 24 7)) (a (.wire 29 35)) := by
  have c1066 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_9 := arithEq_of_rows h (i := 9) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c1066, ← c1067, ← c1068] at e_29_9
  have e_29_10 := arithEq_of_rows h (i := 10) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c1069, ← c1070, ← c1071] at e_29_10
  have hr := e_29_10
  simp only [e_29_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f219 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 51) = bselect (a (.virt 304)) (a (.virt 30)) (a (.virt 34)) := by
  have c1072 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_11 := arithEq_of_rows h (i := 11) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c1072, ← c1073, ← c1074] at e_29_11
  have e_29_12 := arithEq_of_rows h (i := 12) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c1075, ← c1076, ← c1077] at e_29_12
  have hr := e_29_12
  simp only [e_29_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f220 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 29 59) = bselect (a (.virt 306)) (a (.wire 24 15)) (a (.wire 29 51)) := by
  have c1078 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_13 := arithEq_of_rows h (i := 13) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c1078, ← c1079, ← c1080] at e_29_13
  have e_29_14 := arithEq_of_rows h (i := 14) leafCircuit_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c1081, ← c1082, ← c1083] at e_29_14
  have hr := e_29_14
  simp only [e_29_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f221 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) := by
  have c1084 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (leafCircuit_copies34 a h).1
  have c1089 := (leafCircuit_copies34 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_11 := arithEq_of_rows h (i := 11) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c1084, ← c1085, ← c1086] at e_12_11
  have e_21_8 := arithEq_of_rows h (i := 8) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1087, ← c1088, k3, ← c1089] at e_21_8
  have hr := e_21_8
  simp only [e_12_11] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f222 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 7) = bselect (a (.wire 21 35)) (a (.virt 31)) (a (.virt 35)) := by
  have c1090 := (leafCircuit_copies34 a h).2.2.1
  have c1091 := (leafCircuit_copies34 a h).2.2.2.1
  have c1092 := (leafCircuit_copies34 a h).2.2.2.2.1
  have c1093 := (leafCircuit_copies34 a h).2.2.2.2.2.1
  have c1094 := (leafCircuit_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.1
  have e_30_0 := arithEq_of_rows h (i := 0) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c1090, ← c1091, ← c1092] at e_30_0
  have e_30_1 := arithEq_of_rows h (i := 1) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c1093, ← c1094, ← c1095] at e_30_1
  have hr := e_30_1
  simp only [e_30_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f223 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 15) = bselect (a (.virt 308)) (a (.wire 19 51)) (a (.wire 30 7)) := by
  have c1096 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1099 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_2 := arithEq_of_rows h (i := 2) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c1096, ← c1097, ← c1098] at e_30_2
  have e_30_3 := arithEq_of_rows h (i := 3) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c1099, ← c1100, ← c1101] at e_30_3
  have hr := e_30_3
  simp only [e_30_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f224 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) := by
  have c1084 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (leafCircuit_copies34 a h).1
  have c1089 := (leafCircuit_copies34 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_11 := arithEq_of_rows h (i := 11) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c1084, ← c1085, ← c1086] at e_12_11
  have e_21_8 := arithEq_of_rows h (i := 8) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1087, ← c1088, k3, ← c1089] at e_21_8
  have hr := e_21_8
  simp only [e_12_11] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f225 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 23) = bselect (a (.wire 21 35)) (a (.virt 32)) (a (.virt 36)) := by
  have c1102 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1103 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1106 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1107 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_4 := arithEq_of_rows h (i := 4) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c1102, ← c1103, ← c1104] at e_30_4
  have e_30_5 := arithEq_of_rows h (i := 5) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c1105, ← c1106, ← c1107] at e_30_5
  have hr := e_30_5
  simp only [e_30_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f226 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 31) = bselect (a (.virt 308)) (a (.wire 19 59)) (a (.wire 30 23)) := by
  have c1108 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_6 := arithEq_of_rows h (i := 6) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c1108, ← c1109, ← c1110] at e_30_6
  have e_30_7 := arithEq_of_rows h (i := 7) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c1111, ← c1112, ← c1113] at e_30_7
  have hr := e_30_7
  simp only [e_30_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f227 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) := by
  have c1084 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (leafCircuit_copies34 a h).1
  have c1089 := (leafCircuit_copies34 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_11 := arithEq_of_rows h (i := 11) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c1084, ← c1085, ← c1086] at e_12_11
  have e_21_8 := arithEq_of_rows h (i := 8) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1087, ← c1088, k3, ← c1089] at e_21_8
  have hr := e_21_8
  simp only [e_12_11] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f228 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 39) = bselect (a (.wire 21 35)) (a (.virt 33)) (a (.virt 37)) := by
  have c1114 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (leafCircuit_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_30_8 := arithEq_of_rows h (i := 8) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c1114, ← c1115, ← c1116] at e_30_8
  have e_30_9 := arithEq_of_rows h (i := 9) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c1117, ← c1118, ← c1119] at e_30_9
  have hr := e_30_9
  simp only [e_30_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f229 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 47) = bselect (a (.virt 308)) (a (.wire 24 7)) (a (.wire 30 39)) := by
  have c1120 := (leafCircuit_copies35 a h).1
  have c1121 := (leafCircuit_copies35 a h).2.1
  have c1122 := (leafCircuit_copies35 a h).2.2.1
  have c1123 := (leafCircuit_copies35 a h).2.2.2.1
  have c1124 := (leafCircuit_copies35 a h).2.2.2.2.1
  have c1125 := (leafCircuit_copies35 a h).2.2.2.2.2.1
  have e_30_10 := arithEq_of_rows h (i := 10) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c1120, ← c1121, ← c1122] at e_30_10
  have e_30_11 := arithEq_of_rows h (i := 11) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c1123, ← c1124, ← c1125] at e_30_11
  have hr := e_30_11
  simp only [e_30_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f230 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) := by
  have c1084 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (leafCircuit_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (leafCircuit_copies34 a h).1
  have c1089 := (leafCircuit_copies34 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_11 := arithEq_of_rows h (i := 11) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c1084, ← c1085, ← c1086] at e_12_11
  have e_21_8 := arithEq_of_rows h (i := 8) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1087, ← c1088, k3, ← c1089] at e_21_8
  have hr := e_21_8
  simp only [e_12_11] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f231 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 30 55) = bselect (a (.wire 21 35)) (a (.virt 34)) (a (.virt 38)) := by
  have c1126 := (leafCircuit_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_12 := arithEq_of_rows h (i := 12) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c1126, ← c1127, ← c1128] at e_30_12
  have e_30_13 := arithEq_of_rows h (i := 13) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c1129, ← c1130, ← c1131] at e_30_13
  have hr := e_30_13
  simp only [e_30_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f232 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 3) = bselect (a (.virt 308)) (a (.wire 24 15)) (a (.wire 30 55)) := by
  have c1132 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_14 := arithEq_of_rows h (i := 14) leafCircuit_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c1132, ← c1133, ← c1134] at e_30_14
  have e_31_0 := arithEq_of_rows h (i := 0) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c1135, ← c1136, ← c1137] at e_31_0
  have hr := e_31_0
  simp only [e_30_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f233 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 11) = bselect (a (.virt 310)) (a (.wire 19 51)) (a (.virt 35)) := by
  have c1138 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1140 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1143 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_1 := arithEq_of_rows h (i := 1) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c1138, ← c1139, ← c1140] at e_31_1
  have e_31_2 := arithEq_of_rows h (i := 2) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c1141, ← c1142, ← c1143] at e_31_2
  have hr := e_31_2
  simp only [e_31_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f234 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 19) = bselect (a (.virt 310)) (a (.wire 19 59)) (a (.virt 36)) := by
  have c1144 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_3 := arithEq_of_rows h (i := 3) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c1144, ← c1145, ← c1146] at e_31_3
  have e_31_4 := arithEq_of_rows h (i := 4) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c1147, ← c1148, ← c1149] at e_31_4
  have hr := e_31_4
  simp only [e_31_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f235 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 27) = bselect (a (.virt 310)) (a (.wire 24 7)) (a (.virt 37)) := by
  have c1150 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (leafCircuit_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (leafCircuit_copies36 a h).1
  have c1153 := (leafCircuit_copies36 a h).2.1
  have c1154 := (leafCircuit_copies36 a h).2.2.1
  have c1155 := (leafCircuit_copies36 a h).2.2.2.1
  have e_31_5 := arithEq_of_rows h (i := 5) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c1150, ← c1151, ← c1152] at e_31_5
  have e_31_6 := arithEq_of_rows h (i := 6) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c1153, ← c1154, ← c1155] at e_31_6
  have hr := e_31_6
  simp only [e_31_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f236 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 35) = bselect (a (.virt 310)) (a (.wire 24 15)) (a (.virt 38)) := by
  have c1156 := (leafCircuit_copies36 a h).2.2.2.2.1
  have c1157 := (leafCircuit_copies36 a h).2.2.2.2.2.1
  have c1158 := (leafCircuit_copies36 a h).2.2.2.2.2.2.1
  have c1159 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have e_31_7 := arithEq_of_rows h (i := 7) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c1156, ← c1157, ← c1158] at e_31_7
  have e_31_8 := arithEq_of_rows h (i := 8) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c1159, ← c1160, ← c1161] at e_31_8
  have hr := e_31_8
  simp only [e_31_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f237 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 35 12) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 0 ∧ a (.wire 35 13) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 1 ∧ a (.wire 35 14) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 2 ∧ a (.wire 35 15) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 3) := by
  have c1162 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1174 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1177 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1179 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1180 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1182 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1183 := (leafCircuit_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (leafCircuit_copies37 a h).1
  have c1185 := (leafCircuit_copies37 a h).2.1
  have c1186 := (leafCircuit_copies37 a h).2.2.1
  have c1187 := (leafCircuit_copies37 a h).2.2.2.1
  have c1188 := (leafCircuit_copies37 a h).2.2.2.2.1
  have c1189 := (leafCircuit_copies37 a h).2.2.2.2.2.1
  have c1190 := (leafCircuit_copies37 a h).2.2.2.2.2.2.1
  have c1191 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1207 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (leafCircuit_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (leafCircuit_copies38 a h).1
  have c1217 := (leafCircuit_copies38 a h).2.1
  have c1218 := (leafCircuit_copies38 a h).2.2.1
  have c1219 := (leafCircuit_copies38 a h).2.2.2.1
  have c1220 := (leafCircuit_copies38 a h).2.2.2.2.1
  have c1221 := (leafCircuit_copies38 a h).2.2.2.2.2.1
  have c1222 := (leafCircuit_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.1
  have c1224 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_21_9 := arithEq_of_rows h (i := 9) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c1174, ← c1175, k3, ← c1176] at e_21_9
  have e_21_10 := arithEq_of_rows h (i := 10) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c1177, ← c1178, k3, ← c1179] at e_21_10
  have e_21_11 := arithEq_of_rows h (i := 11) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c1180, ← c1181, k3, ← c1182] at e_21_11
  have e_21_12 := arithEq_of_rows h (i := 12) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c1183, ← c1184, k3, ← c1185] at e_21_12
  have e_21_13 := arithEq_of_rows h (i := 13) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c1186, ← c1187, k3, ← c1188] at e_21_13
  have e_21_14 := arithEq_of_rows h (i := 14) leafCircuit_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c1189, ← c1190, k3, ← c1191] at e_21_14
  have e_33_0 := arithEq_of_rows h (i := 0) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c1192, ← c1193, k3, ← c1194] at e_33_0
  have e_33_1 := arithEq_of_rows h (i := 1) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c1195, ← c1196, k3, ← c1197] at e_33_1
  have e_33_2 := arithEq_of_rows h (i := 2) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c1210, ← c1211, k3, ← c1212, k3] at e_33_2
  have hin0 : poseidon2In a 32 = addBlock (fun _ => 0) [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59)] :=
    poseidon2In_first c1162.symm c1163.symm c1164.symm c1165.symm c1166.symm c1167.symm c1168.symm c1169.symm (c1170.symm.trans k0) (c1171.symm.trans k0) (c1172.symm.trans k0) (c1173.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 32) rfl rfl hin0
  have hin1 : poseidon2In a 34 = addBlock (poseidon2Out a 32) [a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] :=
    poseidon2In_chain (by linear_combination -c1198 + e_21_9) (by linear_combination -c1199 + e_21_10) (by linear_combination -c1200 + e_21_11) (by linear_combination -c1201 + e_21_12) (by linear_combination -c1202 + e_21_13) (by linear_combination -c1203 + e_21_14) (by linear_combination -c1204 + e_33_0) (by linear_combination -c1205 + e_33_1) c1206.symm c1207.symm c1208.symm c1209.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 34) rfl rfl hin1
  have hin2 : poseidon2In a 35 = addBlock (poseidon2Out a 34) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c1213 + e_33_2) (c1214.symm.trans (add_zero _).symm) (c1215.symm.trans (add_zero _).symm) (c1216.symm.trans (add_zero _).symm) (c1217.symm.trans (add_zero _).symm) (c1218.symm.trans (add_zero _).symm) (c1219.symm.trans (add_zero _).symm) (c1220.symm.trans (add_zero _).symm) c1221.symm c1222.symm c1223.symm c1224.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 35) rfl rfl hin2
  have hpad : pad10 [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] =
      [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59)] ++ ([a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f238 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 43) = bselect (a (.wire 21 27)) (a (.wire 35 12)) (a (.wire 19 51)) := by
  have c1225 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1227 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_9 := arithEq_of_rows h (i := 9) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c1225, ← c1226, ← c1227] at e_31_9
  have e_31_10 := arithEq_of_rows h (i := 10) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c1228, ← c1229, ← c1230] at e_31_10
  have hr := e_31_10
  simp only [e_31_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f239 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 51) = bselect (a (.wire 21 27)) (a (.wire 35 13)) (a (.wire 19 59)) := by
  have c1231 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1233 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1236 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_11 := arithEq_of_rows h (i := 11) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c1231, ← c1232, ← c1233] at e_31_11
  have e_31_12 := arithEq_of_rows h (i := 12) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c1234, ← c1235, ← c1236] at e_31_12
  have hr := e_31_12
  simp only [e_31_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f240 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 31 59) = bselect (a (.wire 21 27)) (a (.wire 35 14)) (a (.wire 24 7)) := by
  have c1237 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1239 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1241 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_13 := arithEq_of_rows h (i := 13) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c1237, ← c1238, ← c1239] at e_31_13
  have e_31_14 := arithEq_of_rows h (i := 14) leafCircuit_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c1240, ← c1241, ← c1242] at e_31_14
  have hr := e_31_14
  simp only [e_31_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f241 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 7) = bselect (a (.wire 21 27)) (a (.wire 35 15)) (a (.wire 24 15)) := by
  have c1243 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1245 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (leafCircuit_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (leafCircuit_copies39 a h).1
  have e_36_0 := arithEq_of_rows h (i := 0) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_0
  simp only [← c1243, ← c1244, ← c1245] at e_36_0
  have e_36_1 := arithEq_of_rows h (i := 1) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_1
  simp only [← c1246, ← c1247, ← c1248] at e_36_1
  have hr := e_36_1
  simp only [e_36_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f242 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 37 1), a (.wire 37 2), a (.wire 37 3), a (.wire 37 4), a (.wire 37 5)] := by
  have c1249 := (leafCircuit_copies39 a h).2.1
  have c1250 := (leafCircuit_copies39 a h).2.2.1
  have c1251 := (leafCircuit_copies39 a h).2.2.2.1
  have c1252 := (leafCircuit_copies39 a h).2.2.2.2.1
  have c1253 := (leafCircuit_copies39 a h).2.2.2.2.2.1
  have c1254 := (leafCircuit_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.1
  have c1256 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have c1258 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1261 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1264 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1267 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1270 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1271 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1274 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (leafCircuit_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (leafCircuit_copies40 a h).1
  have c1281 := (leafCircuit_copies40 a h).2.1
  have c1282 := (leafCircuit_copies40 a h).2.2.1
  have c1283 := (leafCircuit_copies40 a h).2.2.2.1
  have c1284 := (leafCircuit_copies40 a h).2.2.2.2.1
  have c1285 := (leafCircuit_copies40 a h).2.2.2.2.2.1
  have c1286 := (leafCircuit_copies40 a h).2.2.2.2.2.2.1
  have c1287 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1291 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 37) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1249.trans k0
    · exact c1250.trans k0
    · exact c1251.trans k0
    · exact c1252.trans k0
    · exact c1253.trans k0
    · exact c1254.trans k0
    · exact c1255.trans k0
    · exact c1256.trans k0
    · exact c1257.trans k0
    · exact c1258.trans k0
    · exact c1259.trans k0
    · exact c1260.trans k0
    · exact c1261.trans k0
    · exact c1262.trans k0
    · exact c1263.trans k0
    · exact c1264.trans k0
    · exact c1265.trans k0
    · exact c1266.trans k0
    · exact c1267.trans k0
    · exact c1268.trans k0
    · exact c1269.trans k0
    · exact c1270.trans k0
    · exact c1271.trans k0
    · exact c1272.trans k0
    · exact c1273.trans k0
    · exact c1274.trans k0
    · exact c1275.trans k0
    · exact c1276.trans k0
    · exact c1277.trans k0
    · exact c1278.trans k0
    · exact c1279.trans k0
    · exact c1280.trans k0
    · exact c1281.trans k0
    · exact c1282.trans k0
    · exact c1283.trans k0
    · exact c1284.trans k0
    · exact c1285.trans k0
    · exact c1286.trans k0
    · exact c1287.trans k0
    · exact c1288.trans k0
    · exact c1289.trans k0
    · exact c1290.trans k0
    · exact c1291.trans k0
    · exact c1292.trans k0
    · exact c1293.trans k0
    · exact c1294.trans k0
    · exact c1295.trans k0
    · exact c1296.trans k0
    · exact c1297.trans k0
    · exact c1298.trans k0
    · exact c1299.trans k0
    · exact c1300.trans k0
    · exact c1301.trans k0
    · exact c1302.trans k0
    )
  rwa [c1303] at hr

theorem leafCircuit_f243 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f244 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 5) = band (a (.virt 292)) (a (.wire 37 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f245 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 5) = band (a (.wire 37 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f246 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 5) = bor (a (.virt 14)) (a (.wire 37 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f247 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 37 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f248 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f249 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 5) = a (.virt 14) + a (.wire 37 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f250 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 5) = a (.wire 37 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f251 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 11) = bnot (a (.wire 37 5)) := by
  have c1304 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_2 := arithEq_of_rows h (i := 2) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_2
  simp only [← c1304, k3, ← c1305, k3, ← c1306] at e_36_2
  have hr := e_36_2
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f252 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 11) = band (a (.virt 292)) (a (.wire 36 11)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f253 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f254 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 4) = band (a (.virt 292)) (a (.wire 37 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f255 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 47) = band (a (.wire 37 4)) (a (.wire 36 11)) := by
  have c1307 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1308 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_11 := arithEq_of_rows h (i := 11) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1307, ← c1308, ← c1309] at e_26_11
  have hr := e_26_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f256 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 15) = bor (a (.wire 37 5)) (a (.wire 26 47)) := by
  have c1310 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (leafCircuit_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1312 := (leafCircuit_copies41 a h).1
  have c1313 := (leafCircuit_copies41 a h).2.1
  have c1314 := (leafCircuit_copies41 a h).2.2.1
  have c1315 := (leafCircuit_copies41 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_12 := arithEq_of_rows h (i := 12) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c1310, ← c1311, ← c1312] at e_12_12
  have e_33_3 := arithEq_of_rows h (i := 3) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c1313, ← c1314, k3, ← c1315] at e_33_3
  have hr := e_33_3
  simp only [e_12_12] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f257 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 37 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f258 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f259 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 4) = a (.virt 14) + a (.wire 37 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f260 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 4) = a (.wire 37 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f261 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 15) = bnot (a (.wire 37 4)) := by
  have c1316 := (leafCircuit_copies41 a h).2.2.2.2.1
  have c1317 := (leafCircuit_copies41 a h).2.2.2.2.2.1
  have c1318 := (leafCircuit_copies41 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_3 := arithEq_of_rows h (i := 3) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_3
  simp only [← c1316, k3, ← c1317, k3, ← c1318] at e_36_3
  have hr := e_36_3
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f262 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 51) = band (a (.wire 36 11)) (a (.wire 36 15)) := by
  have c1319 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.1
  have c1321 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have e_26_12 := arithEq_of_rows h (i := 12) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1319, ← c1320, ← c1321] at e_26_12
  have hr := e_26_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f263 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f264 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 3) = band (a (.virt 292)) (a (.wire 37 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f265 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 55) = band (a (.wire 37 3)) (a (.wire 26 51)) := by
  have c1322 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_13 := arithEq_of_rows h (i := 13) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1322, ← c1323, ← c1324] at e_26_13
  have hr := e_26_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f266 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 19) = bor (a (.wire 33 15)) (a (.wire 26 55)) := by
  have c1325 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1329 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_13 := arithEq_of_rows h (i := 13) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c1325, ← c1326, ← c1327] at e_12_13
  have e_33_4 := arithEq_of_rows h (i := 4) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c1328, ← c1329, k3, ← c1330] at e_33_4
  have hr := e_33_4
  simp only [e_12_13] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f267 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 37 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f268 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f269 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 3) = a (.virt 14) + a (.wire 37 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f270 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 3) = a (.wire 37 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f271 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 19) = bnot (a (.wire 37 3)) := by
  have c1331 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_4 := arithEq_of_rows h (i := 4) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_4
  simp only [← c1331, k3, ← c1332, k3, ← c1333] at e_36_4
  have hr := e_36_4
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f272 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 26 59) = band (a (.wire 26 51)) (a (.wire 36 19)) := by
  have c1334 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1335 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_14 := arithEq_of_rows h (i := 14) leafCircuit_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1334, ← c1335, ← c1336] at e_26_14
  have hr := e_26_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f273 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f274 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 37 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f275 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 26 59)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f276 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 19) = bor (a (.wire 33 19)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f277 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 2) = a (.virt 292) * a (.wire 37 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f278 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 3) = a (.virt 294) * a (.wire 37 2) := by
  have c1337 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_38_0 := arithEq_of_rows h (i := 0) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_0
  simp only [← c1337, k5, ← c1338, ← c1339, k5] at e_38_0
  have hr := e_38_0
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f279 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 23) = a (.virt 292) + a (.wire 37 2) := by
  have c1340 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1342 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_33_5 := arithEq_of_rows h (i := 5) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c1340, k3, ← c1341, k3, ← c1342] at e_33_5
  have hr := e_33_5
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f280 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 23) = a (.wire 33 23) - a (.wire 38 3) := by
  have c1343 := (leafCircuit_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (leafCircuit_copies42 a h).1
  have c1345 := (leafCircuit_copies42 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_5 := arithEq_of_rows h (i := 5) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_5
  simp only [← c1343, ← c1344, k3, ← c1345] at e_36_5
  have hr := e_36_5
  linear_combination hr

theorem leafCircuit_f281 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 27) = bnot (a (.wire 36 23)) := by
  have c1346 := (leafCircuit_copies42 a h).2.2.1
  have c1347 := (leafCircuit_copies42 a h).2.2.2.1
  have c1348 := (leafCircuit_copies42 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_6 := arithEq_of_rows h (i := 6) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_6
  simp only [← c1346, k3, ← c1347, k3, ← c1348] at e_36_6
  have hr := e_36_6
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f282 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 7) = band (a (.wire 26 59)) (a (.wire 36 27)) := by
  have c1349 := (leafCircuit_copies42 a h).2.2.2.2.2.1
  have c1350 := (leafCircuit_copies42 a h).2.2.2.2.2.2.1
  have c1351 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.1
  have e_38_1 := arithEq_of_rows h (i := 1) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_1
  simp only [← c1349, ← c1350, ← c1351] at e_38_1
  have hr := e_38_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f283 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f284 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 1) = band (a (.virt 292)) (a (.wire 37 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f285 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 11) = band (a (.wire 37 1)) (a (.wire 38 7)) := by
  have c1352 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have c1354 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_38_2 := arithEq_of_rows h (i := 2) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_2
  simp only [← c1352, ← c1353, ← c1354] at e_38_2
  have hr := e_38_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f286 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 27) = bor (a (.wire 33 19)) (a (.wire 38 11)) := by
  have c1355 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1358 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_12_14 := arithEq_of_rows h (i := 14) leafCircuit_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c1355, ← c1356, ← c1357] at e_12_14
  have e_33_6 := arithEq_of_rows h (i := 6) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c1358, ← c1359, k3, ← c1360] at e_33_6
  have hr := e_33_6
  simp only [e_12_14] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f287 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 37 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f288 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f289 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 1) = a (.virt 14) + a (.wire 37 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f290 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 37 1) = a (.wire 37 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f291 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 36 31) = bnot (a (.wire 37 1)) := by
  have c1361 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1362 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1363 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_7 := arithEq_of_rows h (i := 7) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_7
  simp only [← c1361, k3, ← c1362, k3, ← c1363] at e_36_7
  have hr := e_36_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f292 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 15) = band (a (.wire 38 7)) (a (.wire 36 31)) := by
  have c1364 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_3 := arithEq_of_rows h (i := 3) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_3
  simp only [← c1364, ← c1365, ← c1366] at e_38_3
  have hr := e_38_3
  simp only [band]
  linear_combination hr

theorem leafCircuit_f293 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 209)) 2 := by
  have c1367 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1375 := (leafCircuit_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (leafCircuit_copies43 a h).1
  have c1377 := (leafCircuit_copies43 a h).2.1
  have c1378 := (leafCircuit_copies43 a h).2.2.1
  have c1379 := (leafCircuit_copies43 a h).2.2.2.1
  have c1380 := (leafCircuit_copies43 a h).2.2.2.2.1
  have c1381 := (leafCircuit_copies43 a h).2.2.2.2.2.1
  have c1382 := (leafCircuit_copies43 a h).2.2.2.2.2.2.1
  have c1383 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have c1386 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1389 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1392 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (leafCircuit_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (leafCircuit_copies44 a h).1
  have c1409 := (leafCircuit_copies44 a h).2.1
  have c1410 := (leafCircuit_copies44 a h).2.2.1
  have c1411 := (leafCircuit_copies44 a h).2.2.2.1
  have c1412 := (leafCircuit_copies44 a h).2.2.2.2.1
  have c1413 := (leafCircuit_copies44 a h).2.2.2.2.2.1
  have c1414 := (leafCircuit_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.1
  have c1417 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1423 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 39) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1367.trans k0
    · exact c1368.trans k0
    · exact c1369.trans k0
    · exact c1370.trans k0
    · exact c1371.trans k0
    · exact c1372.trans k0
    · exact c1373.trans k0
    · exact c1374.trans k0
    · exact c1375.trans k0
    · exact c1376.trans k0
    · exact c1377.trans k0
    · exact c1378.trans k0
    · exact c1379.trans k0
    · exact c1380.trans k0
    · exact c1381.trans k0
    · exact c1382.trans k0
    · exact c1383.trans k0
    · exact c1384.trans k0
    · exact c1385.trans k0
    · exact c1386.trans k0
    · exact c1387.trans k0
    · exact c1388.trans k0
    · exact c1389.trans k0
    · exact c1390.trans k0
    · exact c1391.trans k0
    · exact c1392.trans k0
    · exact c1393.trans k0
    · exact c1394.trans k0
    · exact c1395.trans k0
    · exact c1396.trans k0
    · exact c1397.trans k0
    · exact c1398.trans k0
    · exact c1399.trans k0
    · exact c1400.trans k0
    · exact c1401.trans k0
    · exact c1402.trans k0
    · exact c1403.trans k0
    · exact c1404.trans k0
    · exact c1405.trans k0
    · exact c1406.trans k0
    · exact c1407.trans k0
    · exact c1408.trans k0
    · exact c1409.trans k0
    · exact c1410.trans k0
    · exact c1411.trans k0
    · exact c1412.trans k0
    · exact c1413.trans k0
    · exact c1414.trans k0
    · exact c1415.trans k0
    · exact c1416.trans k0
    · exact c1417.trans k0
    · exact c1418.trans k0
    · exact c1419.trans k0
    · exact c1420.trans k0
    · exact c1421.trans k0
    · exact c1422.trans k0
    · exact c1423.trans k0
    )
  rwa [c1424] at hr

theorem leafCircuit_f294 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 209)) (a (.virt 14)) (a (.virt 312)) (a (.virt 313)) := by
  have c1425 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1426 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1428 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1429 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1431 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1432 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1434 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1435 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1437 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1438 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_8 := arithEq_of_rows h (i := 8) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_8
  simp only [← c1425, k3, ← c1426, k3, ← c1427] at e_36_8
  have e_36_9 := arithEq_of_rows h (i := 9) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_9
  simp only [← c1434, ← c1435, k3, ← c1436] at e_36_9
  have e_38_4 := arithEq_of_rows h (i := 4) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_4
  simp only [← c1428, ← c1429, ← c1430] at e_38_4
  have e_38_5 := arithEq_of_rows h (i := 5) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_5
  simp only [← c1431, ← c1432, ← c1433] at e_38_5
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_38_4
    linear_combination c1437.trans k0 - hc
  · have hc := e_36_9
    simp only [e_36_8, e_38_5] at hc
    linear_combination c1438.trans k0 - hc

theorem leafCircuit_f295 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 209)) (a (.virt 292)) (a (.virt 314)) (a (.virt 315)) := by
  have c1439 := (leafCircuit_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1440 := (leafCircuit_copies45 a h).1
  have c1441 := (leafCircuit_copies45 a h).2.1
  have c1442 := (leafCircuit_copies45 a h).2.2.1
  have c1443 := (leafCircuit_copies45 a h).2.2.2.1
  have c1444 := (leafCircuit_copies45 a h).2.2.2.2.1
  have c1445 := (leafCircuit_copies45 a h).2.2.2.2.2.1
  have c1446 := (leafCircuit_copies45 a h).2.2.2.2.2.2.1
  have c1447 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.1
  have c1448 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1451 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1454 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_10 := arithEq_of_rows h (i := 10) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_10
  simp only [← c1439, k3, ← c1440, k3, ← c1441] at e_36_10
  have e_36_11 := arithEq_of_rows h (i := 11) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_11
  simp only [← c1442, ← c1443, k3, ← c1444, k3] at e_36_11
  have e_36_12 := arithEq_of_rows h (i := 12) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_12
  simp only [← c1451, ← c1452, k3, ← c1453] at e_36_12
  have e_38_6 := arithEq_of_rows h (i := 6) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_6
  simp only [← c1445, ← c1446, ← c1447] at e_38_6
  have e_38_7 := arithEq_of_rows h (i := 7) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_7
  simp only [← c1448, ← c1449, ← c1450] at e_38_7
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_38_6
    simp only [e_36_11] at hc
    linear_combination c1454.trans k0 - hc
  · have hc := e_36_12
    simp only [e_36_10, e_38_7, e_36_11] at hc
    linear_combination c1455.trans k0 - hc

theorem leafCircuit_f296 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 209)) (a (.virt 294)) (a (.virt 316)) (a (.virt 317)) := by
  have c1456 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1459 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1465 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1468 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1469 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (leafCircuit_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (leafCircuit_copies46 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_36_13 := arithEq_of_rows h (i := 13) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_13
  simp only [← c1456, k3, ← c1457, k3, ← c1458] at e_36_13
  have e_36_14 := arithEq_of_rows h (i := 14) leafCircuit_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_14
  simp only [← c1459, ← c1460, k3, ← c1461, k5] at e_36_14
  have e_38_8 := arithEq_of_rows h (i := 8) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_8
  simp only [← c1462, ← c1463, ← c1464] at e_38_8
  have e_38_9 := arithEq_of_rows h (i := 9) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_9
  simp only [← c1465, ← c1466, ← c1467] at e_38_9
  have e_40_0 := arithEq_of_rows h (i := 0) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_0
  simp only [← c1468, ← c1469, k3, ← c1470] at e_40_0
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_38_8
    simp only [e_36_14] at hc
    linear_combination c1471.trans k0 - hc
  · have hc := e_40_0
    simp only [e_36_13, e_38_9, e_36_14] at hc
    linear_combination c1472.trans k0 - hc

theorem leafCircuit_f297 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 209)) (a (.virt 295)) (a (.virt 318)) (a (.virt 319)) := by
  have c1473 := (leafCircuit_copies46 a h).2.1
  have c1474 := (leafCircuit_copies46 a h).2.2.1
  have c1475 := (leafCircuit_copies46 a h).2.2.2.1
  have c1476 := (leafCircuit_copies46 a h).2.2.2.2.1
  have c1477 := (leafCircuit_copies46 a h).2.2.2.2.2.1
  have c1478 := (leafCircuit_copies46 a h).2.2.2.2.2.2.1
  have c1479 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.1
  have c1482 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1486 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_38_10 := arithEq_of_rows h (i := 10) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_10
  simp only [← c1479, ← c1480, ← c1481] at e_38_10
  have e_38_11 := arithEq_of_rows h (i := 11) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_11
  simp only [← c1482, ← c1483, ← c1484] at e_38_11
  have e_40_1 := arithEq_of_rows h (i := 1) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_1
  simp only [← c1473, k3, ← c1474, k3, ← c1475] at e_40_1
  have e_40_2 := arithEq_of_rows h (i := 2) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_2
  simp only [← c1476, ← c1477, k3, ← c1478, k6] at e_40_2
  have e_40_3 := arithEq_of_rows h (i := 3) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_3
  simp only [← c1485, ← c1486, k3, ← c1487] at e_40_3
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_38_10
    simp only [e_40_2] at hc
    linear_combination c1488.trans k0 - hc
  · have hc := e_40_3
    simp only [e_40_1, e_38_11, e_40_2] at hc
    linear_combination c1489.trans k0 - hc

theorem leafCircuit_f298 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 40 23) = bselect (a (.virt 312)) (a (.wire 31 43)) (a (.virt 39)) := by
  have c1490 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1493 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_4 := arithEq_of_rows h (i := 4) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_4
  simp only [← c1490, ← c1491, ← c1492] at e_40_4
  have e_40_5 := arithEq_of_rows h (i := 5) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_5
  simp only [← c1493, ← c1494, ← c1495] at e_40_5
  have hr := e_40_5
  simp only [e_40_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f299 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 40 31) = bselect (a (.virt 312)) (a (.wire 31 51)) (a (.virt 40)) := by
  have c1496 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1499 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_6 := arithEq_of_rows h (i := 6) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_6
  simp only [← c1496, ← c1497, ← c1498] at e_40_6
  have e_40_7 := arithEq_of_rows h (i := 7) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_7
  simp only [← c1499, ← c1500, ← c1501] at e_40_7
  have hr := e_40_7
  simp only [e_40_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f300 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 40 39) = bselect (a (.virt 312)) (a (.wire 31 59)) (a (.virt 41)) := by
  have c1502 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1503 := (leafCircuit_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (leafCircuit_copies47 a h).1
  have c1505 := (leafCircuit_copies47 a h).2.1
  have c1506 := (leafCircuit_copies47 a h).2.2.1
  have c1507 := (leafCircuit_copies47 a h).2.2.2.1
  have e_40_8 := arithEq_of_rows h (i := 8) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_8
  simp only [← c1502, ← c1503, ← c1504] at e_40_8
  have e_40_9 := arithEq_of_rows h (i := 9) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_9
  simp only [← c1505, ← c1506, ← c1507] at e_40_9
  have hr := e_40_9
  simp only [e_40_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f301 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 40 47) = bselect (a (.virt 312)) (a (.wire 36 7)) (a (.virt 42)) := by
  have c1508 := (leafCircuit_copies47 a h).2.2.2.2.1
  have c1509 := (leafCircuit_copies47 a h).2.2.2.2.2.1
  have c1510 := (leafCircuit_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.1
  have c1513 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have e_40_10 := arithEq_of_rows h (i := 10) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_10
  simp only [← c1508, ← c1509, ← c1510] at e_40_10
  have e_40_11 := arithEq_of_rows h (i := 11) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_11
  simp only [← c1511, ← c1512, ← c1513] at e_40_11
  have hr := e_40_11
  simp only [e_40_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f302 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 40 55) = bselect (a (.virt 312)) (a (.virt 39)) (a (.virt 43)) := by
  have c1514 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1516 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1519 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_12 := arithEq_of_rows h (i := 12) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_12
  simp only [← c1514, ← c1515, ← c1516] at e_40_12
  have e_40_13 := arithEq_of_rows h (i := 13) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_13
  simp only [← c1517, ← c1518, ← c1519] at e_40_13
  have hr := e_40_13
  simp only [e_40_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f303 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 3) = bselect (a (.virt 314)) (a (.wire 31 43)) (a (.wire 40 55)) := by
  have c1520 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1522 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1523 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_14 := arithEq_of_rows h (i := 14) leafCircuit_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_14
  simp only [← c1520, ← c1521, ← c1522] at e_40_14
  have e_41_0 := arithEq_of_rows h (i := 0) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_0
  simp only [← c1523, ← c1524, ← c1525] at e_41_0
  have hr := e_41_0
  simp only [e_40_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f304 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 11) = bselect (a (.virt 312)) (a (.virt 40)) (a (.virt 44)) := by
  have c1526 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1530 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1531 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_1 := arithEq_of_rows h (i := 1) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_1
  simp only [← c1526, ← c1527, ← c1528] at e_41_1
  have e_41_2 := arithEq_of_rows h (i := 2) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_2
  simp only [← c1529, ← c1530, ← c1531] at e_41_2
  have hr := e_41_2
  simp only [e_41_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f305 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 19) = bselect (a (.virt 314)) (a (.wire 31 51)) (a (.wire 41 11)) := by
  have c1532 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1533 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1535 := (leafCircuit_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (leafCircuit_copies48 a h).1
  have c1537 := (leafCircuit_copies48 a h).2.1
  have e_41_3 := arithEq_of_rows h (i := 3) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_3
  simp only [← c1532, ← c1533, ← c1534] at e_41_3
  have e_41_4 := arithEq_of_rows h (i := 4) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_4
  simp only [← c1535, ← c1536, ← c1537] at e_41_4
  have hr := e_41_4
  simp only [e_41_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f306 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 27) = bselect (a (.virt 312)) (a (.virt 41)) (a (.virt 45)) := by
  have c1538 := (leafCircuit_copies48 a h).2.2.1
  have c1539 := (leafCircuit_copies48 a h).2.2.2.1
  have c1540 := (leafCircuit_copies48 a h).2.2.2.2.1
  have c1541 := (leafCircuit_copies48 a h).2.2.2.2.2.1
  have c1542 := (leafCircuit_copies48 a h).2.2.2.2.2.2.1
  have c1543 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.1
  have e_41_5 := arithEq_of_rows h (i := 5) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_5
  simp only [← c1538, ← c1539, ← c1540] at e_41_5
  have e_41_6 := arithEq_of_rows h (i := 6) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_6
  simp only [← c1541, ← c1542, ← c1543] at e_41_6
  have hr := e_41_6
  simp only [e_41_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f307 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 35) = bselect (a (.virt 314)) (a (.wire 31 59)) (a (.wire 41 27)) := by
  have c1544 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.1
  have c1545 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have c1546 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1547 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1548 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_7 := arithEq_of_rows h (i := 7) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_7
  simp only [← c1544, ← c1545, ← c1546] at e_41_7
  have e_41_8 := arithEq_of_rows h (i := 8) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_8
  simp only [← c1547, ← c1548, ← c1549] at e_41_8
  have hr := e_41_8
  simp only [e_41_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f308 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 43) = bselect (a (.virt 312)) (a (.virt 42)) (a (.virt 46)) := by
  have c1550 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1553 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1554 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_9 := arithEq_of_rows h (i := 9) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_9
  simp only [← c1550, ← c1551, ← c1552] at e_41_9
  have e_41_10 := arithEq_of_rows h (i := 10) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_10
  simp only [← c1553, ← c1554, ← c1555] at e_41_10
  have hr := e_41_10
  simp only [e_41_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f309 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 51) = bselect (a (.virt 314)) (a (.wire 36 7)) (a (.wire 41 43)) := by
  have c1556 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1557 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1558 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1559 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1561 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_11 := arithEq_of_rows h (i := 11) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_11
  simp only [← c1556, ← c1557, ← c1558] at e_41_11
  have e_41_12 := arithEq_of_rows h (i := 12) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_12
  simp only [← c1559, ← c1560, ← c1561] at e_41_12
  have hr := e_41_12
  simp only [e_41_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f310 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) := by
  have c1562 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_33_7 := arithEq_of_rows h (i := 7) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1565, ← c1566, k3, ← c1567] at e_33_7
  have e_42_0 := arithEq_of_rows h (i := 0) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c1562, ← c1563, ← c1564] at e_42_0
  have hr := e_33_7
  simp only [e_42_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f311 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 41 59) = bselect (a (.wire 33 31)) (a (.virt 43)) (a (.virt 47)) := by
  have c1568 := (leafCircuit_copies49 a h).1
  have c1569 := (leafCircuit_copies49 a h).2.1
  have c1570 := (leafCircuit_copies49 a h).2.2.1
  have c1571 := (leafCircuit_copies49 a h).2.2.2.1
  have c1572 := (leafCircuit_copies49 a h).2.2.2.2.1
  have c1573 := (leafCircuit_copies49 a h).2.2.2.2.2.1
  have e_41_13 := arithEq_of_rows h (i := 13) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_13
  simp only [← c1568, ← c1569, ← c1570] at e_41_13
  have e_41_14 := arithEq_of_rows h (i := 14) leafCircuit_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_14
  simp only [← c1571, ← c1572, ← c1573] at e_41_14
  have hr := e_41_14
  simp only [e_41_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f312 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 7) = bselect (a (.virt 316)) (a (.wire 31 43)) (a (.wire 41 59)) := by
  have c1574 := (leafCircuit_copies49 a h).2.2.2.2.2.2.1
  have c1575 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have c1578 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_0 := arithEq_of_rows h (i := 0) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_0
  simp only [← c1574, ← c1575, ← c1576] at e_43_0
  have e_43_1 := arithEq_of_rows h (i := 1) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_1
  simp only [← c1577, ← c1578, ← c1579] at e_43_1
  have hr := e_43_1
  simp only [e_43_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f313 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) := by
  have c1562 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_33_7 := arithEq_of_rows h (i := 7) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1565, ← c1566, k3, ← c1567] at e_33_7
  have e_42_0 := arithEq_of_rows h (i := 0) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c1562, ← c1563, ← c1564] at e_42_0
  have hr := e_33_7
  simp only [e_42_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f314 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 15) = bselect (a (.wire 33 31)) (a (.virt 44)) (a (.virt 48)) := by
  have c1580 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1581 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1583 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1584 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_2 := arithEq_of_rows h (i := 2) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_2
  simp only [← c1580, ← c1581, ← c1582] at e_43_2
  have e_43_3 := arithEq_of_rows h (i := 3) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_3
  simp only [← c1583, ← c1584, ← c1585] at e_43_3
  have hr := e_43_3
  simp only [e_43_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f315 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 23) = bselect (a (.virt 316)) (a (.wire 31 51)) (a (.wire 43 15)) := by
  have c1586 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1590 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_4 := arithEq_of_rows h (i := 4) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_4
  simp only [← c1586, ← c1587, ← c1588] at e_43_4
  have e_43_5 := arithEq_of_rows h (i := 5) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_5
  simp only [← c1589, ← c1590, ← c1591] at e_43_5
  have hr := e_43_5
  simp only [e_43_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f316 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) := by
  have c1562 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_33_7 := arithEq_of_rows h (i := 7) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1565, ← c1566, k3, ← c1567] at e_33_7
  have e_42_0 := arithEq_of_rows h (i := 0) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c1562, ← c1563, ← c1564] at e_42_0
  have hr := e_33_7
  simp only [e_42_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f317 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 31) = bselect (a (.wire 33 31)) (a (.virt 45)) (a (.virt 49)) := by
  have c1592 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1593 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1595 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1597 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_6 := arithEq_of_rows h (i := 6) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_6
  simp only [← c1592, ← c1593, ← c1594] at e_43_6
  have e_43_7 := arithEq_of_rows h (i := 7) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_7
  simp only [← c1595, ← c1596, ← c1597] at e_43_7
  have hr := e_43_7
  simp only [e_43_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f318 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 39) = bselect (a (.virt 316)) (a (.wire 31 59)) (a (.wire 43 31)) := by
  have c1598 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1599 := (leafCircuit_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (leafCircuit_copies50 a h).1
  have c1601 := (leafCircuit_copies50 a h).2.1
  have c1602 := (leafCircuit_copies50 a h).2.2.1
  have c1603 := (leafCircuit_copies50 a h).2.2.2.1
  have e_43_8 := arithEq_of_rows h (i := 8) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_8
  simp only [← c1598, ← c1599, ← c1600] at e_43_8
  have e_43_9 := arithEq_of_rows h (i := 9) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_9
  simp only [← c1601, ← c1602, ← c1603] at e_43_9
  have hr := e_43_9
  simp only [e_43_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f319 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) := by
  have c1562 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (leafCircuit_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_33_7 := arithEq_of_rows h (i := 7) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1565, ← c1566, k3, ← c1567] at e_33_7
  have e_42_0 := arithEq_of_rows h (i := 0) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c1562, ← c1563, ← c1564] at e_42_0
  have hr := e_33_7
  simp only [e_42_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f320 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 47) = bselect (a (.wire 33 31)) (a (.virt 46)) (a (.virt 50)) := by
  have c1604 := (leafCircuit_copies50 a h).2.2.2.2.1
  have c1605 := (leafCircuit_copies50 a h).2.2.2.2.2.1
  have c1606 := (leafCircuit_copies50 a h).2.2.2.2.2.2.1
  have c1607 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.1
  have c1609 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have e_43_10 := arithEq_of_rows h (i := 10) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_10
  simp only [← c1604, ← c1605, ← c1606] at e_43_10
  have e_43_11 := arithEq_of_rows h (i := 11) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_11
  simp only [← c1607, ← c1608, ← c1609] at e_43_11
  have hr := e_43_11
  simp only [e_43_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f321 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 43 55) = bselect (a (.virt 316)) (a (.wire 36 7)) (a (.wire 43 47)) := by
  have c1610 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1612 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1613 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1614 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_12 := arithEq_of_rows h (i := 12) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_12
  simp only [← c1610, ← c1611, ← c1612] at e_43_12
  have e_43_13 := arithEq_of_rows h (i := 13) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_13
  simp only [← c1613, ← c1614, ← c1615] at e_43_13
  have hr := e_43_13
  simp only [e_43_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f322 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 3) = bselect (a (.virt 318)) (a (.wire 31 43)) (a (.virt 47)) := by
  have c1616 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1617 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1619 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1620 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_14 := arithEq_of_rows h (i := 14) leafCircuit_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_14
  simp only [← c1616, ← c1617, ← c1618] at e_43_14
  have e_44_0 := arithEq_of_rows h (i := 0) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_0
  simp only [← c1619, ← c1620, ← c1621] at e_44_0
  have hr := e_44_0
  simp only [e_43_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f323 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 11) = bselect (a (.virt 318)) (a (.wire 31 51)) (a (.virt 48)) := by
  have c1622 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1623 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1626 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_1 := arithEq_of_rows h (i := 1) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_1
  simp only [← c1622, ← c1623, ← c1624] at e_44_1
  have e_44_2 := arithEq_of_rows h (i := 2) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_2
  simp only [← c1625, ← c1626, ← c1627] at e_44_2
  have hr := e_44_2
  simp only [e_44_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f324 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 19) = bselect (a (.virt 318)) (a (.wire 31 59)) (a (.virt 49)) := by
  have c1628 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1629 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1630 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1631 := (leafCircuit_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (leafCircuit_copies51 a h).1
  have c1633 := (leafCircuit_copies51 a h).2.1
  have e_44_3 := arithEq_of_rows h (i := 3) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_3
  simp only [← c1628, ← c1629, ← c1630] at e_44_3
  have e_44_4 := arithEq_of_rows h (i := 4) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_4
  simp only [← c1631, ← c1632, ← c1633] at e_44_4
  have hr := e_44_4
  simp only [e_44_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f325 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 27) = bselect (a (.virt 318)) (a (.wire 36 7)) (a (.virt 50)) := by
  have c1634 := (leafCircuit_copies51 a h).2.2.1
  have c1635 := (leafCircuit_copies51 a h).2.2.2.1
  have c1636 := (leafCircuit_copies51 a h).2.2.2.2.1
  have c1637 := (leafCircuit_copies51 a h).2.2.2.2.2.1
  have c1638 := (leafCircuit_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.1
  have e_44_5 := arithEq_of_rows h (i := 5) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_5
  simp only [← c1634, ← c1635, ← c1636] at e_44_5
  have e_44_6 := arithEq_of_rows h (i := 6) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_6
  simp only [← c1637, ← c1638, ← c1639] at e_44_6
  have hr := e_44_6
  simp only [e_44_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f326 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 48 12) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 0 ∧ a (.wire 48 13) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 1 ∧ a (.wire 48 14) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 2 ∧ a (.wire 48 15) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 3) := by
  have c1640 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1643 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1646 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1649 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1655 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1657 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1658 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1660 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1661 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1663 := (leafCircuit_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (leafCircuit_copies52 a h).1
  have c1665 := (leafCircuit_copies52 a h).2.1
  have c1666 := (leafCircuit_copies52 a h).2.2.1
  have c1667 := (leafCircuit_copies52 a h).2.2.2.1
  have c1668 := (leafCircuit_copies52 a h).2.2.2.2.1
  have c1669 := (leafCircuit_copies52 a h).2.2.2.2.2.1
  have c1670 := (leafCircuit_copies52 a h).2.2.2.2.2.2.1
  have c1671 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.1
  have c1674 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1677 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1679 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1685 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1688 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1689 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1691 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1692 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1694 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (leafCircuit_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (leafCircuit_copies53 a h).1
  have c1697 := (leafCircuit_copies53 a h).2.1
  have c1698 := (leafCircuit_copies53 a h).2.2.1
  have c1699 := (leafCircuit_copies53 a h).2.2.2.1
  have c1700 := (leafCircuit_copies53 a h).2.2.2.2.1
  have c1701 := (leafCircuit_copies53 a h).2.2.2.2.2.1
  have c1702 := (leafCircuit_copies53 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_33_8 := arithEq_of_rows h (i := 8) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c1652, ← c1653, k3, ← c1654] at e_33_8
  have e_33_9 := arithEq_of_rows h (i := 9) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c1655, ← c1656, k3, ← c1657] at e_33_9
  have e_33_10 := arithEq_of_rows h (i := 10) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c1658, ← c1659, k3, ← c1660] at e_33_10
  have e_33_11 := arithEq_of_rows h (i := 11) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c1661, ← c1662, k3, ← c1663] at e_33_11
  have e_33_12 := arithEq_of_rows h (i := 12) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c1664, ← c1665, k3, ← c1666] at e_33_12
  have e_33_13 := arithEq_of_rows h (i := 13) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c1667, ← c1668, k3, ← c1669] at e_33_13
  have e_33_14 := arithEq_of_rows h (i := 14) leafCircuit_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c1670, ← c1671, k3, ← c1672] at e_33_14
  have e_46_0 := arithEq_of_rows h (i := 0) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_0
  simp only [← c1673, ← c1674, k3, ← c1675] at e_46_0
  have e_46_1 := arithEq_of_rows h (i := 1) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_1
  simp only [← c1688, ← c1689, k3, ← c1690, k3] at e_46_1
  have hin0 : poseidon2In a 45 = addBlock (fun _ => 0) [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51)] :=
    poseidon2In_first c1640.symm c1641.symm c1642.symm c1643.symm c1644.symm c1645.symm c1646.symm c1647.symm (c1648.symm.trans k0) (c1649.symm.trans k0) (c1650.symm.trans k0) (c1651.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 45) rfl rfl hin0
  have hin1 : poseidon2In a 47 = addBlock (poseidon2Out a 45) [a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] :=
    poseidon2In_chain (by linear_combination -c1676 + e_33_8) (by linear_combination -c1677 + e_33_9) (by linear_combination -c1678 + e_33_10) (by linear_combination -c1679 + e_33_11) (by linear_combination -c1680 + e_33_12) (by linear_combination -c1681 + e_33_13) (by linear_combination -c1682 + e_33_14) (by linear_combination -c1683 + e_46_0) c1684.symm c1685.symm c1686.symm c1687.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 47) rfl rfl hin1
  have hin2 : poseidon2In a 48 = addBlock (poseidon2Out a 47) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c1691 + e_46_1) (c1692.symm.trans (add_zero _).symm) (c1693.symm.trans (add_zero _).symm) (c1694.symm.trans (add_zero _).symm) (c1695.symm.trans (add_zero _).symm) (c1696.symm.trans (add_zero _).symm) (c1697.symm.trans (add_zero _).symm) (c1698.symm.trans (add_zero _).symm) c1699.symm c1700.symm c1701.symm c1702.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 48) rfl rfl hin2
  have hpad : pad10 [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] =
      [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51)] ++ ([a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f327 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 35) = bselect (a (.wire 33 27)) (a (.wire 48 12)) (a (.wire 31 43)) := by
  have c1703 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.1
  have c1705 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1708 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_7 := arithEq_of_rows h (i := 7) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_7
  simp only [← c1703, ← c1704, ← c1705] at e_44_7
  have e_44_8 := arithEq_of_rows h (i := 8) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_8
  simp only [← c1706, ← c1707, ← c1708] at e_44_8
  have hr := e_44_8
  simp only [e_44_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f328 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 43) = bselect (a (.wire 33 27)) (a (.wire 48 13)) (a (.wire 31 51)) := by
  have c1709 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1711 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1714 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_9 := arithEq_of_rows h (i := 9) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_9
  simp only [← c1709, ← c1710, ← c1711] at e_44_9
  have e_44_10 := arithEq_of_rows h (i := 10) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_10
  simp only [← c1712, ← c1713, ← c1714] at e_44_10
  have hr := e_44_10
  simp only [e_44_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f329 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 51) = bselect (a (.wire 33 27)) (a (.wire 48 14)) (a (.wire 31 59)) := by
  have c1715 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1716 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1717 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1718 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1719 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1720 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_11 := arithEq_of_rows h (i := 11) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_11
  simp only [← c1715, ← c1716, ← c1717] at e_44_11
  have e_44_12 := arithEq_of_rows h (i := 12) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_12
  simp only [← c1718, ← c1719, ← c1720] at e_44_12
  have hr := e_44_12
  simp only [e_44_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f330 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 44 59) = bselect (a (.wire 33 27)) (a (.wire 48 15)) (a (.wire 36 7)) := by
  have c1721 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1722 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1723 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1724 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1725 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1726 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_13 := arithEq_of_rows h (i := 13) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_13
  simp only [← c1721, ← c1722, ← c1723] at e_44_13
  have e_44_14 := arithEq_of_rows h (i := 14) leafCircuit_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_14
  simp only [← c1724, ← c1725, ← c1726] at e_44_14
  have hr := e_44_14
  simp only [e_44_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f331 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 49 1), a (.wire 49 2), a (.wire 49 3), a (.wire 49 4), a (.wire 49 5)] := by
  have c1727 := (leafCircuit_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1728 := (leafCircuit_copies54 a h).1
  have c1729 := (leafCircuit_copies54 a h).2.1
  have c1730 := (leafCircuit_copies54 a h).2.2.1
  have c1731 := (leafCircuit_copies54 a h).2.2.2.1
  have c1732 := (leafCircuit_copies54 a h).2.2.2.2.1
  have c1733 := (leafCircuit_copies54 a h).2.2.2.2.2.1
  have c1734 := (leafCircuit_copies54 a h).2.2.2.2.2.2.1
  have c1735 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.1
  have c1736 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.1
  have c1737 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.1
  have c1738 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1739 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1740 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1741 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1742 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1743 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1744 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1745 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1746 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1747 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1748 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1749 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1750 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1751 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1752 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1753 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1754 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1755 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1756 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1757 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1758 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1759 := (leafCircuit_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1760 := (leafCircuit_copies55 a h).1
  have c1761 := (leafCircuit_copies55 a h).2.1
  have c1762 := (leafCircuit_copies55 a h).2.2.1
  have c1763 := (leafCircuit_copies55 a h).2.2.2.1
  have c1764 := (leafCircuit_copies55 a h).2.2.2.2.1
  have c1765 := (leafCircuit_copies55 a h).2.2.2.2.2.1
  have c1766 := (leafCircuit_copies55 a h).2.2.2.2.2.2.1
  have c1767 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.1
  have c1768 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.1
  have c1769 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.1
  have c1770 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1771 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1772 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1773 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1774 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1775 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1776 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1777 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1778 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1779 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1780 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1781 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 49) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1727.trans k0
    · exact c1728.trans k0
    · exact c1729.trans k0
    · exact c1730.trans k0
    · exact c1731.trans k0
    · exact c1732.trans k0
    · exact c1733.trans k0
    · exact c1734.trans k0
    · exact c1735.trans k0
    · exact c1736.trans k0
    · exact c1737.trans k0
    · exact c1738.trans k0
    · exact c1739.trans k0
    · exact c1740.trans k0
    · exact c1741.trans k0
    · exact c1742.trans k0
    · exact c1743.trans k0
    · exact c1744.trans k0
    · exact c1745.trans k0
    · exact c1746.trans k0
    · exact c1747.trans k0
    · exact c1748.trans k0
    · exact c1749.trans k0
    · exact c1750.trans k0
    · exact c1751.trans k0
    · exact c1752.trans k0
    · exact c1753.trans k0
    · exact c1754.trans k0
    · exact c1755.trans k0
    · exact c1756.trans k0
    · exact c1757.trans k0
    · exact c1758.trans k0
    · exact c1759.trans k0
    · exact c1760.trans k0
    · exact c1761.trans k0
    · exact c1762.trans k0
    · exact c1763.trans k0
    · exact c1764.trans k0
    · exact c1765.trans k0
    · exact c1766.trans k0
    · exact c1767.trans k0
    · exact c1768.trans k0
    · exact c1769.trans k0
    · exact c1770.trans k0
    · exact c1771.trans k0
    · exact c1772.trans k0
    · exact c1773.trans k0
    · exact c1774.trans k0
    · exact c1775.trans k0
    · exact c1776.trans k0
    · exact c1777.trans k0
    · exact c1778.trans k0
    · exact c1779.trans k0
    · exact c1780.trans k0
    )
  rwa [c1781] at hr

theorem leafCircuit_f332 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f333 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 5) = band (a (.virt 292)) (a (.wire 49 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f334 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 5) = band (a (.wire 49 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f335 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 5) = bor (a (.virt 14)) (a (.wire 49 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f336 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 49 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f337 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f338 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 5) = a (.virt 14) + a (.wire 49 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f339 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 5) = a (.wire 49 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f340 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 3) = bnot (a (.wire 49 5)) := by
  have c1782 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1783 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1784 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_0 := arithEq_of_rows h (i := 0) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_0
  simp only [← c1782, k3, ← c1783, k3, ← c1784] at e_50_0
  have hr := e_50_0
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f341 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 3) = band (a (.virt 292)) (a (.wire 50 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f342 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f343 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 4) = band (a (.virt 292)) (a (.wire 49 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f344 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 51) = band (a (.wire 49 4)) (a (.wire 50 3)) := by
  have c1785 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1786 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1787 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_12 := arithEq_of_rows h (i := 12) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_12
  simp only [← c1785, ← c1786, ← c1787] at e_38_12
  have hr := e_38_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f345 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 11) = bor (a (.wire 49 5)) (a (.wire 38 51)) := by
  have c1788 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1789 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1790 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1791 := (leafCircuit_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1792 := (leafCircuit_copies56 a h).1
  have c1793 := (leafCircuit_copies56 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_1 := arithEq_of_rows h (i := 1) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_1
  simp only [← c1788, ← c1789, ← c1790] at e_42_1
  have e_46_2 := arithEq_of_rows h (i := 2) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_2
  simp only [← c1791, ← c1792, k3, ← c1793] at e_46_2
  have hr := e_46_2
  simp only [e_42_1] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f346 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 49 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f347 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f348 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 4) = a (.virt 14) + a (.wire 49 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f349 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 4) = a (.wire 49 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f350 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 7) = bnot (a (.wire 49 4)) := by
  have c1794 := (leafCircuit_copies56 a h).2.2.1
  have c1795 := (leafCircuit_copies56 a h).2.2.2.1
  have c1796 := (leafCircuit_copies56 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_1 := arithEq_of_rows h (i := 1) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_1
  simp only [← c1794, k3, ← c1795, k3, ← c1796] at e_50_1
  have hr := e_50_1
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f351 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 55) = band (a (.wire 50 3)) (a (.wire 50 7)) := by
  have c1797 := (leafCircuit_copies56 a h).2.2.2.2.2.1
  have c1798 := (leafCircuit_copies56 a h).2.2.2.2.2.2.1
  have c1799 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.1
  have e_38_13 := arithEq_of_rows h (i := 13) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_13
  simp only [← c1797, ← c1798, ← c1799] at e_38_13
  have hr := e_38_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f352 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f353 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 3) = band (a (.virt 292)) (a (.wire 49 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f354 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 38 59) = band (a (.wire 49 3)) (a (.wire 38 55)) := by
  have c1800 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.1
  have c1801 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.1
  have c1802 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_38_14 := arithEq_of_rows h (i := 14) leafCircuit_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_14
  simp only [← c1800, ← c1801, ← c1802] at e_38_14
  have hr := e_38_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f355 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 15) = bor (a (.wire 46 11)) (a (.wire 38 59)) := by
  have c1803 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1804 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1805 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1806 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1807 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1808 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_2 := arithEq_of_rows h (i := 2) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_2
  simp only [← c1803, ← c1804, ← c1805] at e_42_2
  have e_46_3 := arithEq_of_rows h (i := 3) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_3
  simp only [← c1806, ← c1807, k3, ← c1808] at e_46_3
  have hr := e_46_3
  simp only [e_42_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f356 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 49 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f357 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f358 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 3) = a (.virt 14) + a (.wire 49 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f359 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 3) = a (.wire 49 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f360 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 11) = bnot (a (.wire 49 3)) := by
  have c1809 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1810 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1811 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_2 := arithEq_of_rows h (i := 2) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_2
  simp only [← c1809, k3, ← c1810, k3, ← c1811] at e_50_2
  have hr := e_50_2
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f361 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 3) = band (a (.wire 38 55)) (a (.wire 50 11)) := by
  have c1812 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1813 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1814 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_51_0 := arithEq_of_rows h (i := 0) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_0
  simp only [← c1812, ← c1813, ← c1814] at e_51_0
  have hr := e_51_0
  simp only [band]
  linear_combination hr

theorem leafCircuit_f362 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f363 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 49 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f364 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 51 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f365 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 15) = bor (a (.wire 46 15)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f366 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 2) = a (.virt 292) * a (.wire 49 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f367 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 7) = a (.virt 294) * a (.wire 49 2) := by
  have c1815 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1816 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1817 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_51_1 := arithEq_of_rows h (i := 1) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_1
  simp only [← c1815, k5, ← c1816, ← c1817, k5] at e_51_1
  have hr := e_51_1
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f368 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 19) = a (.virt 292) + a (.wire 49 2) := by
  have c1818 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1819 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1820 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_46_4 := arithEq_of_rows h (i := 4) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_4
  simp only [← c1818, k3, ← c1819, k3, ← c1820] at e_46_4
  have hr := e_46_4
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f369 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 15) = a (.wire 46 19) - a (.wire 51 7) := by
  have c1821 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1822 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1823 := (leafCircuit_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_3 := arithEq_of_rows h (i := 3) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_3
  simp only [← c1821, ← c1822, k3, ← c1823] at e_50_3
  have hr := e_50_3
  linear_combination hr

theorem leafCircuit_f370 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 19) = bnot (a (.wire 50 15)) := by
  have c1824 := (leafCircuit_copies57 a h).1
  have c1825 := (leafCircuit_copies57 a h).2.1
  have c1826 := (leafCircuit_copies57 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_4 := arithEq_of_rows h (i := 4) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_4
  simp only [← c1824, k3, ← c1825, k3, ← c1826] at e_50_4
  have hr := e_50_4
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f371 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 11) = band (a (.wire 51 3)) (a (.wire 50 19)) := by
  have c1827 := (leafCircuit_copies57 a h).2.2.2.1
  have c1828 := (leafCircuit_copies57 a h).2.2.2.2.1
  have c1829 := (leafCircuit_copies57 a h).2.2.2.2.2.1
  have e_51_2 := arithEq_of_rows h (i := 2) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_2
  simp only [← c1827, ← c1828, ← c1829] at e_51_2
  have hr := e_51_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f372 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f373 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 49 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f374 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 51 11)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f375 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 15) = bor (a (.wire 46 15)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f376 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 49 1) = a (.virt 292) * a (.wire 49 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f377 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 15) = a (.virt 294) * a (.wire 49 1) := by
  have c1830 := (leafCircuit_copies57 a h).2.2.2.2.2.2.1
  have c1831 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.1
  have c1832 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_51_3 := arithEq_of_rows h (i := 3) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_3
  simp only [← c1830, k5, ← c1831, ← c1832, k5] at e_51_3
  have hr := e_51_3
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f378 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 23) = a (.virt 292) + a (.wire 49 1) := by
  have c1833 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.1
  have c1834 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1835 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_46_5 := arithEq_of_rows h (i := 5) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_5
  simp only [← c1833, k3, ← c1834, k3, ← c1835] at e_46_5
  have hr := e_46_5
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f379 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 23) = a (.wire 46 23) - a (.wire 51 15) := by
  have c1836 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1837 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1838 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_5 := arithEq_of_rows h (i := 5) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_5
  simp only [← c1836, ← c1837, k3, ← c1838] at e_50_5
  have hr := e_50_5
  linear_combination hr

theorem leafCircuit_f380 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 50 27) = bnot (a (.wire 50 23)) := by
  have c1839 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1840 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1841 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_6 := arithEq_of_rows h (i := 6) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_6
  simp only [← c1839, k3, ← c1840, k3, ← c1841] at e_50_6
  have hr := e_50_6
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f381 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 19) = band (a (.wire 51 11)) (a (.wire 50 27)) := by
  have c1842 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1843 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1844 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_51_4 := arithEq_of_rows h (i := 4) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_4
  simp only [← c1842, ← c1843, ← c1844] at e_51_4
  have hr := e_51_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f382 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 210)) 2 := by
  have c1845 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1846 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1847 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1848 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1849 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1850 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1851 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1852 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1853 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1854 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1855 := (leafCircuit_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1856 := (leafCircuit_copies58 a h).1
  have c1857 := (leafCircuit_copies58 a h).2.1
  have c1858 := (leafCircuit_copies58 a h).2.2.1
  have c1859 := (leafCircuit_copies58 a h).2.2.2.1
  have c1860 := (leafCircuit_copies58 a h).2.2.2.2.1
  have c1861 := (leafCircuit_copies58 a h).2.2.2.2.2.1
  have c1862 := (leafCircuit_copies58 a h).2.2.2.2.2.2.1
  have c1863 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.1
  have c1864 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.1
  have c1865 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.1
  have c1866 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1867 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1868 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1869 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1870 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1871 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1872 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1873 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1874 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1875 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1876 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1877 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1878 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1879 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1880 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1881 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1882 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1883 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1884 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1885 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1886 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1887 := (leafCircuit_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1888 := (leafCircuit_copies59 a h).1
  have c1889 := (leafCircuit_copies59 a h).2.1
  have c1890 := (leafCircuit_copies59 a h).2.2.1
  have c1891 := (leafCircuit_copies59 a h).2.2.2.1
  have c1892 := (leafCircuit_copies59 a h).2.2.2.2.1
  have c1893 := (leafCircuit_copies59 a h).2.2.2.2.2.1
  have c1894 := (leafCircuit_copies59 a h).2.2.2.2.2.2.1
  have c1895 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.1
  have c1896 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.1
  have c1897 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.1
  have c1898 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1899 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1900 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1901 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1902 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 52) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1845.trans k0
    · exact c1846.trans k0
    · exact c1847.trans k0
    · exact c1848.trans k0
    · exact c1849.trans k0
    · exact c1850.trans k0
    · exact c1851.trans k0
    · exact c1852.trans k0
    · exact c1853.trans k0
    · exact c1854.trans k0
    · exact c1855.trans k0
    · exact c1856.trans k0
    · exact c1857.trans k0
    · exact c1858.trans k0
    · exact c1859.trans k0
    · exact c1860.trans k0
    · exact c1861.trans k0
    · exact c1862.trans k0
    · exact c1863.trans k0
    · exact c1864.trans k0
    · exact c1865.trans k0
    · exact c1866.trans k0
    · exact c1867.trans k0
    · exact c1868.trans k0
    · exact c1869.trans k0
    · exact c1870.trans k0
    · exact c1871.trans k0
    · exact c1872.trans k0
    · exact c1873.trans k0
    · exact c1874.trans k0
    · exact c1875.trans k0
    · exact c1876.trans k0
    · exact c1877.trans k0
    · exact c1878.trans k0
    · exact c1879.trans k0
    · exact c1880.trans k0
    · exact c1881.trans k0
    · exact c1882.trans k0
    · exact c1883.trans k0
    · exact c1884.trans k0
    · exact c1885.trans k0
    · exact c1886.trans k0
    · exact c1887.trans k0
    · exact c1888.trans k0
    · exact c1889.trans k0
    · exact c1890.trans k0
    · exact c1891.trans k0
    · exact c1892.trans k0
    · exact c1893.trans k0
    · exact c1894.trans k0
    · exact c1895.trans k0
    · exact c1896.trans k0
    · exact c1897.trans k0
    · exact c1898.trans k0
    · exact c1899.trans k0
    · exact c1900.trans k0
    · exact c1901.trans k0
    )
  rwa [c1902] at hr

theorem leafCircuit_f383 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 210)) (a (.virt 14)) (a (.virt 320)) (a (.virt 321)) := by
  have c1903 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1904 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1905 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1906 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1907 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1908 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1909 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1910 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1911 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1912 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1913 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1914 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1915 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1916 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_7 := arithEq_of_rows h (i := 7) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_7
  simp only [← c1903, k3, ← c1904, k3, ← c1905] at e_50_7
  have e_50_8 := arithEq_of_rows h (i := 8) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_8
  simp only [← c1912, ← c1913, k3, ← c1914] at e_50_8
  have e_51_5 := arithEq_of_rows h (i := 5) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_5
  simp only [← c1906, ← c1907, ← c1908] at e_51_5
  have e_51_6 := arithEq_of_rows h (i := 6) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_6
  simp only [← c1909, ← c1910, ← c1911] at e_51_6
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_51_5
    linear_combination c1915.trans k0 - hc
  · have hc := e_50_8
    simp only [e_50_7, e_51_6] at hc
    linear_combination c1916.trans k0 - hc

theorem leafCircuit_f384 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 210)) (a (.virt 292)) (a (.virt 322)) (a (.virt 323)) := by
  have c1917 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1918 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1919 := (leafCircuit_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1920 := (leafCircuit_copies60 a h).1
  have c1921 := (leafCircuit_copies60 a h).2.1
  have c1922 := (leafCircuit_copies60 a h).2.2.1
  have c1923 := (leafCircuit_copies60 a h).2.2.2.1
  have c1924 := (leafCircuit_copies60 a h).2.2.2.2.1
  have c1925 := (leafCircuit_copies60 a h).2.2.2.2.2.1
  have c1926 := (leafCircuit_copies60 a h).2.2.2.2.2.2.1
  have c1927 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.1
  have c1928 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.1
  have c1929 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.1
  have c1930 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1931 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1932 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1933 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_9 := arithEq_of_rows h (i := 9) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_9
  simp only [← c1917, k3, ← c1918, k3, ← c1919] at e_50_9
  have e_50_10 := arithEq_of_rows h (i := 10) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_10
  simp only [← c1920, ← c1921, k3, ← c1922, k3] at e_50_10
  have e_50_11 := arithEq_of_rows h (i := 11) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_11
  simp only [← c1929, ← c1930, k3, ← c1931] at e_50_11
  have e_51_7 := arithEq_of_rows h (i := 7) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_7
  simp only [← c1923, ← c1924, ← c1925] at e_51_7
  have e_51_8 := arithEq_of_rows h (i := 8) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_8
  simp only [← c1926, ← c1927, ← c1928] at e_51_8
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_51_7
    simp only [e_50_10] at hc
    linear_combination c1932.trans k0 - hc
  · have hc := e_50_11
    simp only [e_50_9, e_51_8, e_50_10] at hc
    linear_combination c1933.trans k0 - hc

theorem leafCircuit_f385 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 210)) (a (.virt 294)) (a (.virt 324)) (a (.virt 325)) := by
  have c1934 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1935 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1936 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1937 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1938 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1939 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1940 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1941 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1942 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1943 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1944 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1945 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1946 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1947 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1948 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1949 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1950 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_50_12 := arithEq_of_rows h (i := 12) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_12
  simp only [← c1934, k3, ← c1935, k3, ← c1936] at e_50_12
  have e_50_13 := arithEq_of_rows h (i := 13) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_13
  simp only [← c1937, ← c1938, k3, ← c1939, k5] at e_50_13
  have e_50_14 := arithEq_of_rows h (i := 14) leafCircuit_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_14
  simp only [← c1946, ← c1947, k3, ← c1948] at e_50_14
  have e_51_9 := arithEq_of_rows h (i := 9) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_9
  simp only [← c1940, ← c1941, ← c1942] at e_51_9
  have e_51_10 := arithEq_of_rows h (i := 10) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_10
  simp only [← c1943, ← c1944, ← c1945] at e_51_10
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_51_9
    simp only [e_50_13] at hc
    linear_combination c1949.trans k0 - hc
  · have hc := e_50_14
    simp only [e_50_12, e_51_10, e_50_13] at hc
    linear_combination c1950.trans k0 - hc

theorem leafCircuit_f386 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 210)) (a (.virt 295)) (a (.virt 326)) (a (.virt 327)) := by
  have c1951 := (leafCircuit_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1952 := (leafCircuit_copies61 a h).1
  have c1953 := (leafCircuit_copies61 a h).2.1
  have c1954 := (leafCircuit_copies61 a h).2.2.1
  have c1955 := (leafCircuit_copies61 a h).2.2.2.1
  have c1956 := (leafCircuit_copies61 a h).2.2.2.2.1
  have c1957 := (leafCircuit_copies61 a h).2.2.2.2.2.1
  have c1958 := (leafCircuit_copies61 a h).2.2.2.2.2.2.1
  have c1959 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.1
  have c1960 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.1
  have c1961 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.1
  have c1962 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1963 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1964 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1965 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1966 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1967 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_51_11 := arithEq_of_rows h (i := 11) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_11
  simp only [← c1957, ← c1958, ← c1959] at e_51_11
  have e_51_12 := arithEq_of_rows h (i := 12) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_12
  simp only [← c1960, ← c1961, ← c1962] at e_51_12
  have e_53_0 := arithEq_of_rows h (i := 0) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_0
  simp only [← c1951, k3, ← c1952, k3, ← c1953] at e_53_0
  have e_53_1 := arithEq_of_rows h (i := 1) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_1
  simp only [← c1954, ← c1955, k3, ← c1956, k6] at e_53_1
  have e_53_2 := arithEq_of_rows h (i := 2) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_2
  simp only [← c1963, ← c1964, k3, ← c1965] at e_53_2
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_51_11
    simp only [e_53_1] at hc
    linear_combination c1966.trans k0 - hc
  · have hc := e_53_2
    simp only [e_53_0, e_51_12, e_53_1] at hc
    linear_combination c1967.trans k0 - hc

theorem leafCircuit_f387 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 53 19) = bselect (a (.virt 320)) (a (.wire 44 35)) (a (.virt 51)) := by
  have c1968 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1969 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1970 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1971 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1972 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1973 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_3 := arithEq_of_rows h (i := 3) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_3
  simp only [← c1968, ← c1969, ← c1970] at e_53_3
  have e_53_4 := arithEq_of_rows h (i := 4) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_4
  simp only [← c1971, ← c1972, ← c1973] at e_53_4
  have hr := e_53_4
  simp only [e_53_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f388 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 53 27) = bselect (a (.virt 320)) (a (.wire 44 43)) (a (.virt 52)) := by
  have c1974 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1975 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1976 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1977 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1978 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1979 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_5 := arithEq_of_rows h (i := 5) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_5
  simp only [← c1974, ← c1975, ← c1976] at e_53_5
  have e_53_6 := arithEq_of_rows h (i := 6) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_6
  simp only [← c1977, ← c1978, ← c1979] at e_53_6
  have hr := e_53_6
  simp only [e_53_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f389 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 53 35) = bselect (a (.virt 320)) (a (.wire 44 51)) (a (.virt 53)) := by
  have c1980 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1981 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1982 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1983 := (leafCircuit_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1984 := (leafCircuit_copies62 a h).1
  have c1985 := (leafCircuit_copies62 a h).2.1
  have e_53_7 := arithEq_of_rows h (i := 7) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_7
  simp only [← c1980, ← c1981, ← c1982] at e_53_7
  have e_53_8 := arithEq_of_rows h (i := 8) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_8
  simp only [← c1983, ← c1984, ← c1985] at e_53_8
  have hr := e_53_8
  simp only [e_53_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f390 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 53 43) = bselect (a (.virt 320)) (a (.wire 44 59)) (a (.virt 54)) := by
  have c1986 := (leafCircuit_copies62 a h).2.2.1
  have c1987 := (leafCircuit_copies62 a h).2.2.2.1
  have c1988 := (leafCircuit_copies62 a h).2.2.2.2.1
  have c1989 := (leafCircuit_copies62 a h).2.2.2.2.2.1
  have c1990 := (leafCircuit_copies62 a h).2.2.2.2.2.2.1
  have c1991 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.1
  have e_53_9 := arithEq_of_rows h (i := 9) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_9
  simp only [← c1986, ← c1987, ← c1988] at e_53_9
  have e_53_10 := arithEq_of_rows h (i := 10) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_10
  simp only [← c1989, ← c1990, ← c1991] at e_53_10
  have hr := e_53_10
  simp only [e_53_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f391 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 53 51) = bselect (a (.virt 320)) (a (.virt 51)) (a (.virt 55)) := by
  have c1992 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.1
  have c1993 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.1
  have c1994 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1995 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1996 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1997 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_11 := arithEq_of_rows h (i := 11) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_11
  simp only [← c1992, ← c1993, ← c1994] at e_53_11
  have e_53_12 := arithEq_of_rows h (i := 12) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_12
  simp only [← c1995, ← c1996, ← c1997] at e_53_12
  have hr := e_53_12
  simp only [e_53_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f392 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 53 59) = bselect (a (.virt 322)) (a (.wire 44 35)) (a (.wire 53 51)) := by
  have c1998 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1999 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2000 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2001 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2002 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2003 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_13 := arithEq_of_rows h (i := 13) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_13
  simp only [← c1998, ← c1999, ← c2000] at e_53_13
  have e_53_14 := arithEq_of_rows h (i := 14) leafCircuit_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_14
  simp only [← c2001, ← c2002, ← c2003] at e_53_14
  have hr := e_53_14
  simp only [e_53_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f393 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 7) = bselect (a (.virt 320)) (a (.virt 52)) (a (.virt 56)) := by
  have c2004 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2005 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2006 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2007 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2008 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2009 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_0 := arithEq_of_rows h (i := 0) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_0
  simp only [← c2004, ← c2005, ← c2006] at e_54_0
  have e_54_1 := arithEq_of_rows h (i := 1) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_1
  simp only [← c2007, ← c2008, ← c2009] at e_54_1
  have hr := e_54_1
  simp only [e_54_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f394 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 15) = bselect (a (.virt 322)) (a (.wire 44 43)) (a (.wire 54 7)) := by
  have c2010 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2011 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2012 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2013 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2014 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2015 := (leafCircuit_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_54_2 := arithEq_of_rows h (i := 2) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_2
  simp only [← c2010, ← c2011, ← c2012] at e_54_2
  have e_54_3 := arithEq_of_rows h (i := 3) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_3
  simp only [← c2013, ← c2014, ← c2015] at e_54_3
  have hr := e_54_3
  simp only [e_54_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f395 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 23) = bselect (a (.virt 320)) (a (.virt 53)) (a (.virt 57)) := by
  have c2016 := (leafCircuit_copies63 a h).1
  have c2017 := (leafCircuit_copies63 a h).2.1
  have c2018 := (leafCircuit_copies63 a h).2.2.1
  have c2019 := (leafCircuit_copies63 a h).2.2.2.1
  have c2020 := (leafCircuit_copies63 a h).2.2.2.2.1
  have c2021 := (leafCircuit_copies63 a h).2.2.2.2.2.1
  have e_54_4 := arithEq_of_rows h (i := 4) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_4
  simp only [← c2016, ← c2017, ← c2018] at e_54_4
  have e_54_5 := arithEq_of_rows h (i := 5) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_5
  simp only [← c2019, ← c2020, ← c2021] at e_54_5
  have hr := e_54_5
  simp only [e_54_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f396 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 31) = bselect (a (.virt 322)) (a (.wire 44 51)) (a (.wire 54 23)) := by
  have c2022 := (leafCircuit_copies63 a h).2.2.2.2.2.2.1
  have c2023 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.1
  have c2024 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.1
  have c2025 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.1
  have c2026 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2027 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_6 := arithEq_of_rows h (i := 6) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_6
  simp only [← c2022, ← c2023, ← c2024] at e_54_6
  have e_54_7 := arithEq_of_rows h (i := 7) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_7
  simp only [← c2025, ← c2026, ← c2027] at e_54_7
  have hr := e_54_7
  simp only [e_54_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f397 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 39) = bselect (a (.virt 320)) (a (.virt 54)) (a (.virt 58)) := by
  have c2028 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2029 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2030 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2031 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2032 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2033 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_8 := arithEq_of_rows h (i := 8) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_8
  simp only [← c2028, ← c2029, ← c2030] at e_54_8
  have e_54_9 := arithEq_of_rows h (i := 9) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_9
  simp only [← c2031, ← c2032, ← c2033] at e_54_9
  have hr := e_54_9
  simp only [e_54_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f398 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 47) = bselect (a (.virt 322)) (a (.wire 44 59)) (a (.wire 54 39)) := by
  have c2034 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2035 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2036 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2037 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2038 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2039 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_10 := arithEq_of_rows h (i := 10) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_10
  simp only [← c2034, ← c2035, ← c2036] at e_54_10
  have e_54_11 := arithEq_of_rows h (i := 11) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_11
  simp only [← c2037, ← c2038, ← c2039] at e_54_11
  have hr := e_54_11
  simp only [e_54_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f399 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) := by
  have c2040 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_3 := arithEq_of_rows h (i := 3) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c2040, ← c2041, ← c2042] at e_42_3
  have e_46_6 := arithEq_of_rows h (i := 6) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c2043, ← c2044, k3, ← c2045] at e_46_6
  have hr := e_46_6
  simp only [e_42_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f400 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 54 55) = bselect (a (.wire 46 27)) (a (.virt 55)) (a (.virt 59)) := by
  have c2046 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2047 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2048 := (leafCircuit_copies64 a h).1
  have c2049 := (leafCircuit_copies64 a h).2.1
  have c2050 := (leafCircuit_copies64 a h).2.2.1
  have c2051 := (leafCircuit_copies64 a h).2.2.2.1
  have e_54_12 := arithEq_of_rows h (i := 12) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_12
  simp only [← c2046, ← c2047, ← c2048] at e_54_12
  have e_54_13 := arithEq_of_rows h (i := 13) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_13
  simp only [← c2049, ← c2050, ← c2051] at e_54_13
  have hr := e_54_13
  simp only [e_54_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f401 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 3) = bselect (a (.virt 324)) (a (.wire 44 35)) (a (.wire 54 55)) := by
  have c2052 := (leafCircuit_copies64 a h).2.2.2.2.1
  have c2053 := (leafCircuit_copies64 a h).2.2.2.2.2.1
  have c2054 := (leafCircuit_copies64 a h).2.2.2.2.2.2.1
  have c2055 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.1
  have c2056 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.1
  have c2057 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.1
  have e_54_14 := arithEq_of_rows h (i := 14) leafCircuit_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_14
  simp only [← c2052, ← c2053, ← c2054] at e_54_14
  have e_55_0 := arithEq_of_rows h (i := 0) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_0
  simp only [← c2055, ← c2056, ← c2057] at e_55_0
  have hr := e_55_0
  simp only [e_54_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f402 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) := by
  have c2040 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_3 := arithEq_of_rows h (i := 3) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c2040, ← c2041, ← c2042] at e_42_3
  have e_46_6 := arithEq_of_rows h (i := 6) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c2043, ← c2044, k3, ← c2045] at e_46_6
  have hr := e_46_6
  simp only [e_42_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f403 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 11) = bselect (a (.wire 46 27)) (a (.virt 56)) (a (.virt 60)) := by
  have c2058 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2059 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2060 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2061 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2062 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2063 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_1 := arithEq_of_rows h (i := 1) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_1
  simp only [← c2058, ← c2059, ← c2060] at e_55_1
  have e_55_2 := arithEq_of_rows h (i := 2) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_2
  simp only [← c2061, ← c2062, ← c2063] at e_55_2
  have hr := e_55_2
  simp only [e_55_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f404 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 19) = bselect (a (.virt 324)) (a (.wire 44 43)) (a (.wire 55 11)) := by
  have c2064 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2065 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2066 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2067 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2068 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2069 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_3 := arithEq_of_rows h (i := 3) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_3
  simp only [← c2064, ← c2065, ← c2066] at e_55_3
  have e_55_4 := arithEq_of_rows h (i := 4) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_4
  simp only [← c2067, ← c2068, ← c2069] at e_55_4
  have hr := e_55_4
  simp only [e_55_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f405 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) := by
  have c2040 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_3 := arithEq_of_rows h (i := 3) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c2040, ← c2041, ← c2042] at e_42_3
  have e_46_6 := arithEq_of_rows h (i := 6) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c2043, ← c2044, k3, ← c2045] at e_46_6
  have hr := e_46_6
  simp only [e_42_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f406 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 27) = bselect (a (.wire 46 27)) (a (.virt 57)) (a (.virt 61)) := by
  have c2070 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2071 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2072 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2073 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2074 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2075 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_5 := arithEq_of_rows h (i := 5) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_5
  simp only [← c2070, ← c2071, ← c2072] at e_55_5
  have e_55_6 := arithEq_of_rows h (i := 6) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_6
  simp only [← c2073, ← c2074, ← c2075] at e_55_6
  have hr := e_55_6
  simp only [e_55_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f407 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 35) = bselect (a (.virt 324)) (a (.wire 44 51)) (a (.wire 55 27)) := by
  have c2076 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2077 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2078 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2079 := (leafCircuit_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2080 := (leafCircuit_copies65 a h).1
  have c2081 := (leafCircuit_copies65 a h).2.1
  have e_55_7 := arithEq_of_rows h (i := 7) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_7
  simp only [← c2076, ← c2077, ← c2078] at e_55_7
  have e_55_8 := arithEq_of_rows h (i := 8) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_8
  simp only [← c2079, ← c2080, ← c2081] at e_55_8
  have hr := e_55_8
  simp only [e_55_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f408 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) := by
  have c2040 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (leafCircuit_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_3 := arithEq_of_rows h (i := 3) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c2040, ← c2041, ← c2042] at e_42_3
  have e_46_6 := arithEq_of_rows h (i := 6) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c2043, ← c2044, k3, ← c2045] at e_46_6
  have hr := e_46_6
  simp only [e_42_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f409 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 43) = bselect (a (.wire 46 27)) (a (.virt 58)) (a (.virt 62)) := by
  have c2082 := (leafCircuit_copies65 a h).2.2.1
  have c2083 := (leafCircuit_copies65 a h).2.2.2.1
  have c2084 := (leafCircuit_copies65 a h).2.2.2.2.1
  have c2085 := (leafCircuit_copies65 a h).2.2.2.2.2.1
  have c2086 := (leafCircuit_copies65 a h).2.2.2.2.2.2.1
  have c2087 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.1
  have e_55_9 := arithEq_of_rows h (i := 9) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_9
  simp only [← c2082, ← c2083, ← c2084] at e_55_9
  have e_55_10 := arithEq_of_rows h (i := 10) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_10
  simp only [← c2085, ← c2086, ← c2087] at e_55_10
  have hr := e_55_10
  simp only [e_55_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f410 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 51) = bselect (a (.virt 324)) (a (.wire 44 59)) (a (.wire 55 43)) := by
  have c2088 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.1
  have c2089 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.1
  have c2090 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2091 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2092 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2093 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_11 := arithEq_of_rows h (i := 11) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_11
  simp only [← c2088, ← c2089, ← c2090] at e_55_11
  have e_55_12 := arithEq_of_rows h (i := 12) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_12
  simp only [← c2091, ← c2092, ← c2093] at e_55_12
  have hr := e_55_12
  simp only [e_55_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f411 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 55 59) = bselect (a (.virt 326)) (a (.wire 44 35)) (a (.virt 59)) := by
  have c2094 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2095 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2096 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2097 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2098 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2099 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_13 := arithEq_of_rows h (i := 13) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_13
  simp only [← c2094, ← c2095, ← c2096] at e_55_13
  have e_55_14 := arithEq_of_rows h (i := 14) leafCircuit_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_14
  simp only [← c2097, ← c2098, ← c2099] at e_55_14
  have hr := e_55_14
  simp only [e_55_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f412 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 7) = bselect (a (.virt 326)) (a (.wire 44 43)) (a (.virt 60)) := by
  have c2100 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2101 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2102 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2103 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2104 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2105 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_0 := arithEq_of_rows h (i := 0) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_0
  simp only [← c2100, ← c2101, ← c2102] at e_56_0
  have e_56_1 := arithEq_of_rows h (i := 1) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_1
  simp only [← c2103, ← c2104, ← c2105] at e_56_1
  have hr := e_56_1
  simp only [e_56_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f413 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 15) = bselect (a (.virt 326)) (a (.wire 44 51)) (a (.virt 61)) := by
  have c2106 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2107 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2108 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2109 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2110 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2111 := (leafCircuit_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_56_2 := arithEq_of_rows h (i := 2) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_2
  simp only [← c2106, ← c2107, ← c2108] at e_56_2
  have e_56_3 := arithEq_of_rows h (i := 3) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_3
  simp only [← c2109, ← c2110, ← c2111] at e_56_3
  have hr := e_56_3
  simp only [e_56_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f414 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 23) = bselect (a (.virt 326)) (a (.wire 44 59)) (a (.virt 62)) := by
  have c2112 := (leafCircuit_copies66 a h).1
  have c2113 := (leafCircuit_copies66 a h).2.1
  have c2114 := (leafCircuit_copies66 a h).2.2.1
  have c2115 := (leafCircuit_copies66 a h).2.2.2.1
  have c2116 := (leafCircuit_copies66 a h).2.2.2.2.1
  have c2117 := (leafCircuit_copies66 a h).2.2.2.2.2.1
  have e_56_4 := arithEq_of_rows h (i := 4) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_4
  simp only [← c2112, ← c2113, ← c2114] at e_56_4
  have e_56_5 := arithEq_of_rows h (i := 5) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_5
  simp only [← c2115, ← c2116, ← c2117] at e_56_5
  have hr := e_56_5
  simp only [e_56_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f415 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 60 12) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 0 ∧ a (.wire 60 13) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 1 ∧ a (.wire 60 14) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 2 ∧ a (.wire 60 15) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 3) := by
  have c2118 := (leafCircuit_copies66 a h).2.2.2.2.2.2.1
  have c2119 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.1
  have c2120 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.1
  have c2121 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.1
  have c2122 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2123 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2124 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2125 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2126 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2127 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2128 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2129 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2130 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2131 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2132 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2133 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2134 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2135 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2136 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2137 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2138 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2139 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2140 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2141 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2142 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2143 := (leafCircuit_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2144 := (leafCircuit_copies67 a h).1
  have c2145 := (leafCircuit_copies67 a h).2.1
  have c2146 := (leafCircuit_copies67 a h).2.2.1
  have c2147 := (leafCircuit_copies67 a h).2.2.2.1
  have c2148 := (leafCircuit_copies67 a h).2.2.2.2.1
  have c2149 := (leafCircuit_copies67 a h).2.2.2.2.2.1
  have c2150 := (leafCircuit_copies67 a h).2.2.2.2.2.2.1
  have c2151 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.1
  have c2152 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.1
  have c2153 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.1
  have c2154 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2155 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2156 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2157 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2158 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2159 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2160 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2161 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2162 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2163 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2164 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2165 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2166 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2167 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2168 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2169 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2170 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2171 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2172 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2173 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2174 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2175 := (leafCircuit_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2176 := (leafCircuit_copies68 a h).1
  have c2177 := (leafCircuit_copies68 a h).2.1
  have c2178 := (leafCircuit_copies68 a h).2.2.1
  have c2179 := (leafCircuit_copies68 a h).2.2.2.1
  have c2180 := (leafCircuit_copies68 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_46_7 := arithEq_of_rows h (i := 7) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_7
  simp only [← c2130, ← c2131, k3, ← c2132] at e_46_7
  have e_46_8 := arithEq_of_rows h (i := 8) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_8
  simp only [← c2133, ← c2134, k3, ← c2135] at e_46_8
  have e_46_9 := arithEq_of_rows h (i := 9) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_9
  simp only [← c2136, ← c2137, k3, ← c2138] at e_46_9
  have e_46_10 := arithEq_of_rows h (i := 10) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_10
  simp only [← c2139, ← c2140, k3, ← c2141] at e_46_10
  have e_46_11 := arithEq_of_rows h (i := 11) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_11
  simp only [← c2142, ← c2143, k3, ← c2144] at e_46_11
  have e_46_12 := arithEq_of_rows h (i := 12) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_12
  simp only [← c2145, ← c2146, k3, ← c2147] at e_46_12
  have e_46_13 := arithEq_of_rows h (i := 13) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_13
  simp only [← c2148, ← c2149, k3, ← c2150] at e_46_13
  have e_46_14 := arithEq_of_rows h (i := 14) leafCircuit_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_14
  simp only [← c2151, ← c2152, k3, ← c2153] at e_46_14
  have e_59_0 := arithEq_of_rows h (i := 0) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_0
  simp only [← c2166, ← c2167, k3, ← c2168, k3] at e_59_0
  have hin0 : poseidon2In a 57 = addBlock (fun _ => 0) [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47)] :=
    poseidon2In_first c2118.symm c2119.symm c2120.symm c2121.symm c2122.symm c2123.symm c2124.symm c2125.symm (c2126.symm.trans k0) (c2127.symm.trans k0) (c2128.symm.trans k0) (c2129.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 57) rfl rfl hin0
  have hin1 : poseidon2In a 58 = addBlock (poseidon2Out a 57) [a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] :=
    poseidon2In_chain (by linear_combination -c2154 + e_46_7) (by linear_combination -c2155 + e_46_8) (by linear_combination -c2156 + e_46_9) (by linear_combination -c2157 + e_46_10) (by linear_combination -c2158 + e_46_11) (by linear_combination -c2159 + e_46_12) (by linear_combination -c2160 + e_46_13) (by linear_combination -c2161 + e_46_14) c2162.symm c2163.symm c2164.symm c2165.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 58) rfl rfl hin1
  have hin2 : poseidon2In a 60 = addBlock (poseidon2Out a 58) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c2169 + e_59_0) (c2170.symm.trans (add_zero _).symm) (c2171.symm.trans (add_zero _).symm) (c2172.symm.trans (add_zero _).symm) (c2173.symm.trans (add_zero _).symm) (c2174.symm.trans (add_zero _).symm) (c2175.symm.trans (add_zero _).symm) (c2176.symm.trans (add_zero _).symm) c2177.symm c2178.symm c2179.symm c2180.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 60) rfl rfl hin2
  have hpad : pad10 [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] =
      [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47)] ++ ([a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f416 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 31) = bselect (a (.wire 46 15)) (a (.wire 60 12)) (a (.wire 44 35)) := by
  have c2181 := (leafCircuit_copies68 a h).2.2.2.2.2.1
  have c2182 := (leafCircuit_copies68 a h).2.2.2.2.2.2.1
  have c2183 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.1
  have c2184 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.1
  have c2185 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.1
  have c2186 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_56_6 := arithEq_of_rows h (i := 6) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_6
  simp only [← c2181, ← c2182, ← c2183] at e_56_6
  have e_56_7 := arithEq_of_rows h (i := 7) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_7
  simp only [← c2184, ← c2185, ← c2186] at e_56_7
  have hr := e_56_7
  simp only [e_56_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f417 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 39) = bselect (a (.wire 46 15)) (a (.wire 60 13)) (a (.wire 44 43)) := by
  have c2187 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2188 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2189 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2190 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2191 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2192 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_8 := arithEq_of_rows h (i := 8) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_8
  simp only [← c2187, ← c2188, ← c2189] at e_56_8
  have e_56_9 := arithEq_of_rows h (i := 9) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_9
  simp only [← c2190, ← c2191, ← c2192] at e_56_9
  have hr := e_56_9
  simp only [e_56_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f418 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 47) = bselect (a (.wire 46 15)) (a (.wire 60 14)) (a (.wire 44 51)) := by
  have c2193 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2194 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2195 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2196 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2197 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2198 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_10 := arithEq_of_rows h (i := 10) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_10
  simp only [← c2193, ← c2194, ← c2195] at e_56_10
  have e_56_11 := arithEq_of_rows h (i := 11) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_11
  simp only [← c2196, ← c2197, ← c2198] at e_56_11
  have hr := e_56_11
  simp only [e_56_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f419 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 55) = bselect (a (.wire 46 15)) (a (.wire 60 15)) (a (.wire 44 59)) := by
  have c2199 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2200 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2201 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2202 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2203 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2204 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_12 := arithEq_of_rows h (i := 12) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_12
  simp only [← c2199, ← c2200, ← c2201] at e_56_12
  have e_56_13 := arithEq_of_rows h (i := 13) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_13
  simp only [← c2202, ← c2203, ← c2204] at e_56_13
  have hr := e_56_13
  simp only [e_56_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f420 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 61 1), a (.wire 61 2), a (.wire 61 3), a (.wire 61 4), a (.wire 61 5)] := by
  have c2205 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2206 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2207 := (leafCircuit_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2208 := (leafCircuit_copies69 a h).1
  have c2209 := (leafCircuit_copies69 a h).2.1
  have c2210 := (leafCircuit_copies69 a h).2.2.1
  have c2211 := (leafCircuit_copies69 a h).2.2.2.1
  have c2212 := (leafCircuit_copies69 a h).2.2.2.2.1
  have c2213 := (leafCircuit_copies69 a h).2.2.2.2.2.1
  have c2214 := (leafCircuit_copies69 a h).2.2.2.2.2.2.1
  have c2215 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.1
  have c2216 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.1
  have c2217 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.1
  have c2218 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2219 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2220 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2221 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2222 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2223 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2224 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2225 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2226 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2227 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2228 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2229 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2230 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2231 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2232 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2233 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2234 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2235 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2236 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2237 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2238 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2239 := (leafCircuit_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2240 := (leafCircuit_copies70 a h).1
  have c2241 := (leafCircuit_copies70 a h).2.1
  have c2242 := (leafCircuit_copies70 a h).2.2.1
  have c2243 := (leafCircuit_copies70 a h).2.2.2.1
  have c2244 := (leafCircuit_copies70 a h).2.2.2.2.1
  have c2245 := (leafCircuit_copies70 a h).2.2.2.2.2.1
  have c2246 := (leafCircuit_copies70 a h).2.2.2.2.2.2.1
  have c2247 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.1
  have c2248 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.1
  have c2249 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.1
  have c2250 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2251 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2252 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2253 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2254 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2255 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2256 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2257 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2258 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2259 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 61) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2205.trans k0
    · exact c2206.trans k0
    · exact c2207.trans k0
    · exact c2208.trans k0
    · exact c2209.trans k0
    · exact c2210.trans k0
    · exact c2211.trans k0
    · exact c2212.trans k0
    · exact c2213.trans k0
    · exact c2214.trans k0
    · exact c2215.trans k0
    · exact c2216.trans k0
    · exact c2217.trans k0
    · exact c2218.trans k0
    · exact c2219.trans k0
    · exact c2220.trans k0
    · exact c2221.trans k0
    · exact c2222.trans k0
    · exact c2223.trans k0
    · exact c2224.trans k0
    · exact c2225.trans k0
    · exact c2226.trans k0
    · exact c2227.trans k0
    · exact c2228.trans k0
    · exact c2229.trans k0
    · exact c2230.trans k0
    · exact c2231.trans k0
    · exact c2232.trans k0
    · exact c2233.trans k0
    · exact c2234.trans k0
    · exact c2235.trans k0
    · exact c2236.trans k0
    · exact c2237.trans k0
    · exact c2238.trans k0
    · exact c2239.trans k0
    · exact c2240.trans k0
    · exact c2241.trans k0
    · exact c2242.trans k0
    · exact c2243.trans k0
    · exact c2244.trans k0
    · exact c2245.trans k0
    · exact c2246.trans k0
    · exact c2247.trans k0
    · exact c2248.trans k0
    · exact c2249.trans k0
    · exact c2250.trans k0
    · exact c2251.trans k0
    · exact c2252.trans k0
    · exact c2253.trans k0
    · exact c2254.trans k0
    · exact c2255.trans k0
    · exact c2256.trans k0
    · exact c2257.trans k0
    · exact c2258.trans k0
    )
  rwa [c2259] at hr

theorem leafCircuit_f421 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f422 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 5) = band (a (.virt 292)) (a (.wire 61 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f423 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 5) = band (a (.wire 61 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f424 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 5) = bor (a (.virt 14)) (a (.wire 61 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f425 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 61 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f426 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f427 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 5) = a (.virt 14) + a (.wire 61 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f428 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 5) = a (.wire 61 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f429 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 59) = bnot (a (.wire 61 5)) := by
  have c2260 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2261 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2262 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_56_14 := arithEq_of_rows h (i := 14) leafCircuit_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_14
  simp only [← c2260, k3, ← c2261, k3, ← c2262] at e_56_14
  have hr := e_56_14
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f430 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 56 59) = band (a (.virt 292)) (a (.wire 56 59)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f431 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f432 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 4) = band (a (.virt 292)) (a (.wire 61 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f433 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 55) = band (a (.wire 61 4)) (a (.wire 56 59)) := by
  have c2263 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2264 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2265 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_51_13 := arithEq_of_rows h (i := 13) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_13
  simp only [← c2263, ← c2264, ← c2265] at e_51_13
  have hr := e_51_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f434 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 7) = bor (a (.wire 61 5)) (a (.wire 51 55)) := by
  have c2266 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2267 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2268 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2269 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2270 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2271 := (leafCircuit_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_4 := arithEq_of_rows h (i := 4) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_4
  simp only [← c2266, ← c2267, ← c2268] at e_42_4
  have e_59_1 := arithEq_of_rows h (i := 1) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_1
  simp only [← c2269, ← c2270, k3, ← c2271] at e_59_1
  have hr := e_59_1
  simp only [e_42_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f435 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 61 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f436 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f437 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 4) = a (.virt 14) + a (.wire 61 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f438 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 4) = a (.wire 61 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f439 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 62 3) = bnot (a (.wire 61 4)) := by
  have c2272 := (leafCircuit_copies71 a h).1
  have c2273 := (leafCircuit_copies71 a h).2.1
  have c2274 := (leafCircuit_copies71 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_0 := arithEq_of_rows h (i := 0) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_0
  simp only [← c2272, k3, ← c2273, k3, ← c2274] at e_62_0
  have hr := e_62_0
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f440 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 51 59) = band (a (.wire 56 59)) (a (.wire 62 3)) := by
  have c2275 := (leafCircuit_copies71 a h).2.2.2.1
  have c2276 := (leafCircuit_copies71 a h).2.2.2.2.1
  have c2277 := (leafCircuit_copies71 a h).2.2.2.2.2.1
  have e_51_14 := arithEq_of_rows h (i := 14) leafCircuit_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_14
  simp only [← c2275, ← c2276, ← c2277] at e_51_14
  have hr := e_51_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f441 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f442 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 61 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f443 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 51 59)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f444 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 7) = bor (a (.wire 59 7)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f445 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 3) = a (.virt 292) * a (.wire 61 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f446 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 3) = a (.virt 294) * a (.wire 61 3) := by
  have c2278 := (leafCircuit_copies71 a h).2.2.2.2.2.2.1
  have c2279 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.1
  have c2280 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_63_0 := arithEq_of_rows h (i := 0) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_0
  simp only [← c2278, k5, ← c2279, ← c2280, k5] at e_63_0
  have hr := e_63_0
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f447 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 11) = a (.virt 292) + a (.wire 61 3) := by
  have c2281 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.1
  have c2282 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2283 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_59_2 := arithEq_of_rows h (i := 2) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_2
  simp only [← c2281, k3, ← c2282, k3, ← c2283] at e_59_2
  have hr := e_59_2
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f448 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 62 7) = a (.wire 59 11) - a (.wire 63 3) := by
  have c2284 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2285 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2286 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_1 := arithEq_of_rows h (i := 1) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_1
  simp only [← c2284, ← c2285, k3, ← c2286] at e_62_1
  have hr := e_62_1
  linear_combination hr

theorem leafCircuit_f449 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 62 11) = bnot (a (.wire 62 7)) := by
  have c2287 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2288 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2289 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_2 := arithEq_of_rows h (i := 2) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_2
  simp only [← c2287, k3, ← c2288, k3, ← c2289] at e_62_2
  have hr := e_62_2
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f450 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 7) = band (a (.wire 51 59)) (a (.wire 62 11)) := by
  have c2290 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2291 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2292 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_1 := arithEq_of_rows h (i := 1) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_1
  simp only [← c2290, ← c2291, ← c2292] at e_63_1
  have hr := e_63_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f451 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f452 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 2) = band (a (.virt 292)) (a (.wire 61 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f453 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 11) = band (a (.wire 61 2)) (a (.wire 63 7)) := by
  have c2293 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2294 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2295 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_2 := arithEq_of_rows h (i := 2) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_2
  simp only [← c2293, ← c2294, ← c2295] at e_63_2
  have hr := e_63_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f454 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 15) = bor (a (.wire 59 7)) (a (.wire 63 11)) := by
  have c2296 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2297 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2298 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2299 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2300 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2301 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_5 := arithEq_of_rows h (i := 5) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_5
  simp only [← c2296, ← c2297, ← c2298] at e_42_5
  have e_59_3 := arithEq_of_rows h (i := 3) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_3
  simp only [← c2299, ← c2300, k3, ← c2301] at e_59_3
  have hr := e_59_3
  simp only [e_42_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f455 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 61 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f456 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f457 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 2) = a (.virt 14) + a (.wire 61 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f458 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 2) = a (.wire 61 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f459 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 62 15) = bnot (a (.wire 61 2)) := by
  have c2302 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2303 := (leafCircuit_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2304 := (leafCircuit_copies72 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_3 := arithEq_of_rows h (i := 3) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_3
  simp only [← c2302, k3, ← c2303, k3, ← c2304] at e_62_3
  have hr := e_62_3
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f460 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 15) = band (a (.wire 63 7)) (a (.wire 62 15)) := by
  have c2305 := (leafCircuit_copies72 a h).2.1
  have c2306 := (leafCircuit_copies72 a h).2.2.1
  have c2307 := (leafCircuit_copies72 a h).2.2.2.1
  have e_63_3 := arithEq_of_rows h (i := 3) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_3
  simp only [← c2305, ← c2306, ← c2307] at e_63_3
  have hr := e_63_3
  simp only [band]
  linear_combination hr

theorem leafCircuit_f461 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f462 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 1) = band (a (.virt 292)) (a (.wire 61 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f463 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 19) = band (a (.wire 61 1)) (a (.wire 63 15)) := by
  have c2308 := (leafCircuit_copies72 a h).2.2.2.2.1
  have c2309 := (leafCircuit_copies72 a h).2.2.2.2.2.1
  have c2310 := (leafCircuit_copies72 a h).2.2.2.2.2.2.1
  have e_63_4 := arithEq_of_rows h (i := 4) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_4
  simp only [← c2308, ← c2309, ← c2310] at e_63_4
  have hr := e_63_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f464 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 19) = bor (a (.wire 59 15)) (a (.wire 63 19)) := by
  have c2311 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.1
  have c2312 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.1
  have c2313 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.1
  have c2314 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2315 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2316 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_6 := arithEq_of_rows h (i := 6) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_6
  simp only [← c2311, ← c2312, ← c2313] at e_42_6
  have e_59_4 := arithEq_of_rows h (i := 4) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_4
  simp only [← c2314, ← c2315, k3, ← c2316] at e_59_4
  have hr := e_59_4
  simp only [e_42_6] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f465 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 61 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f466 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f467 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 1) = a (.virt 14) + a (.wire 61 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f468 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 61 1) = a (.wire 61 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f469 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 62 19) = bnot (a (.wire 61 1)) := by
  have c2317 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2318 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2319 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_4 := arithEq_of_rows h (i := 4) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_4
  simp only [← c2317, k3, ← c2318, k3, ← c2319] at e_62_4
  have hr := e_62_4
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f470 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 23) = band (a (.wire 63 15)) (a (.wire 62 19)) := by
  have c2320 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2321 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2322 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_5 := arithEq_of_rows h (i := 5) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_5
  simp only [← c2320, ← c2321, ← c2322] at e_63_5
  have hr := e_63_5
  simp only [band]
  linear_combination hr

theorem leafCircuit_f471 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 211)) 2 := by
  have c2323 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2324 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2325 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2326 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2327 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2328 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2329 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2330 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2331 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2332 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2333 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2334 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2335 := (leafCircuit_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2336 := (leafCircuit_copies73 a h).1
  have c2337 := (leafCircuit_copies73 a h).2.1
  have c2338 := (leafCircuit_copies73 a h).2.2.1
  have c2339 := (leafCircuit_copies73 a h).2.2.2.1
  have c2340 := (leafCircuit_copies73 a h).2.2.2.2.1
  have c2341 := (leafCircuit_copies73 a h).2.2.2.2.2.1
  have c2342 := (leafCircuit_copies73 a h).2.2.2.2.2.2.1
  have c2343 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.1
  have c2344 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.1
  have c2345 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.1
  have c2346 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2347 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2348 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2349 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2350 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2351 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2352 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2353 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2354 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2355 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2356 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2357 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2358 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2359 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2360 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2361 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2362 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2363 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2364 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2365 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2366 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2367 := (leafCircuit_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2368 := (leafCircuit_copies74 a h).1
  have c2369 := (leafCircuit_copies74 a h).2.1
  have c2370 := (leafCircuit_copies74 a h).2.2.1
  have c2371 := (leafCircuit_copies74 a h).2.2.2.1
  have c2372 := (leafCircuit_copies74 a h).2.2.2.2.1
  have c2373 := (leafCircuit_copies74 a h).2.2.2.2.2.1
  have c2374 := (leafCircuit_copies74 a h).2.2.2.2.2.2.1
  have c2375 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.1
  have c2376 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.1
  have c2377 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.1
  have c2378 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2379 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2380 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 64) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2323.trans k0
    · exact c2324.trans k0
    · exact c2325.trans k0
    · exact c2326.trans k0
    · exact c2327.trans k0
    · exact c2328.trans k0
    · exact c2329.trans k0
    · exact c2330.trans k0
    · exact c2331.trans k0
    · exact c2332.trans k0
    · exact c2333.trans k0
    · exact c2334.trans k0
    · exact c2335.trans k0
    · exact c2336.trans k0
    · exact c2337.trans k0
    · exact c2338.trans k0
    · exact c2339.trans k0
    · exact c2340.trans k0
    · exact c2341.trans k0
    · exact c2342.trans k0
    · exact c2343.trans k0
    · exact c2344.trans k0
    · exact c2345.trans k0
    · exact c2346.trans k0
    · exact c2347.trans k0
    · exact c2348.trans k0
    · exact c2349.trans k0
    · exact c2350.trans k0
    · exact c2351.trans k0
    · exact c2352.trans k0
    · exact c2353.trans k0
    · exact c2354.trans k0
    · exact c2355.trans k0
    · exact c2356.trans k0
    · exact c2357.trans k0
    · exact c2358.trans k0
    · exact c2359.trans k0
    · exact c2360.trans k0
    · exact c2361.trans k0
    · exact c2362.trans k0
    · exact c2363.trans k0
    · exact c2364.trans k0
    · exact c2365.trans k0
    · exact c2366.trans k0
    · exact c2367.trans k0
    · exact c2368.trans k0
    · exact c2369.trans k0
    · exact c2370.trans k0
    · exact c2371.trans k0
    · exact c2372.trans k0
    · exact c2373.trans k0
    · exact c2374.trans k0
    · exact c2375.trans k0
    · exact c2376.trans k0
    · exact c2377.trans k0
    · exact c2378.trans k0
    · exact c2379.trans k0
    )
  rwa [c2380] at hr

theorem leafCircuit_f472 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 211)) (a (.virt 14)) (a (.virt 328)) (a (.virt 329)) := by
  have c2381 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2382 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2383 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2384 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2385 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2386 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2387 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2388 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2389 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2390 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2391 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2392 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2393 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2394 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_5 := arithEq_of_rows h (i := 5) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_5
  simp only [← c2381, k3, ← c2382, k3, ← c2383] at e_62_5
  have e_62_6 := arithEq_of_rows h (i := 6) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_6
  simp only [← c2390, ← c2391, k3, ← c2392] at e_62_6
  have e_63_6 := arithEq_of_rows h (i := 6) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_6
  simp only [← c2384, ← c2385, ← c2386] at e_63_6
  have e_63_7 := arithEq_of_rows h (i := 7) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_7
  simp only [← c2387, ← c2388, ← c2389] at e_63_7
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_63_6
    linear_combination c2393.trans k0 - hc
  · have hc := e_62_6
    simp only [e_62_5, e_63_7] at hc
    linear_combination c2394.trans k0 - hc

theorem leafCircuit_f473 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 211)) (a (.virt 292)) (a (.virt 330)) (a (.virt 331)) := by
  have c2395 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2396 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2397 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2398 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2399 := (leafCircuit_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2400 := (leafCircuit_copies75 a h).1
  have c2401 := (leafCircuit_copies75 a h).2.1
  have c2402 := (leafCircuit_copies75 a h).2.2.1
  have c2403 := (leafCircuit_copies75 a h).2.2.2.1
  have c2404 := (leafCircuit_copies75 a h).2.2.2.2.1
  have c2405 := (leafCircuit_copies75 a h).2.2.2.2.2.1
  have c2406 := (leafCircuit_copies75 a h).2.2.2.2.2.2.1
  have c2407 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.1
  have c2408 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.1
  have c2409 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.1
  have c2410 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2411 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_7 := arithEq_of_rows h (i := 7) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_7
  simp only [← c2395, k3, ← c2396, k3, ← c2397] at e_62_7
  have e_62_8 := arithEq_of_rows h (i := 8) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_8
  simp only [← c2398, ← c2399, k3, ← c2400, k3] at e_62_8
  have e_62_9 := arithEq_of_rows h (i := 9) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_9
  simp only [← c2407, ← c2408, k3, ← c2409] at e_62_9
  have e_63_8 := arithEq_of_rows h (i := 8) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_8
  simp only [← c2401, ← c2402, ← c2403] at e_63_8
  have e_63_9 := arithEq_of_rows h (i := 9) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_9
  simp only [← c2404, ← c2405, ← c2406] at e_63_9
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_63_8
    simp only [e_62_8] at hc
    linear_combination c2410.trans k0 - hc
  · have hc := e_62_9
    simp only [e_62_7, e_63_9, e_62_8] at hc
    linear_combination c2411.trans k0 - hc

theorem leafCircuit_f474 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 211)) (a (.virt 294)) (a (.virt 332)) (a (.virt 333)) := by
  have c2412 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2413 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2414 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2415 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2416 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2417 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2418 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2419 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2420 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2421 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2422 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2423 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2424 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2425 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2426 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2427 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2428 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_10 := arithEq_of_rows h (i := 10) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_10
  simp only [← c2412, k3, ← c2413, k3, ← c2414] at e_62_10
  have e_62_11 := arithEq_of_rows h (i := 11) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_11
  simp only [← c2415, ← c2416, k3, ← c2417, k5] at e_62_11
  have e_62_12 := arithEq_of_rows h (i := 12) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_12
  simp only [← c2424, ← c2425, k3, ← c2426] at e_62_12
  have e_63_10 := arithEq_of_rows h (i := 10) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_10
  simp only [← c2418, ← c2419, ← c2420] at e_63_10
  have e_63_11 := arithEq_of_rows h (i := 11) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_11
  simp only [← c2421, ← c2422, ← c2423] at e_63_11
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_63_10
    simp only [e_62_11] at hc
    linear_combination c2427.trans k0 - hc
  · have hc := e_62_12
    simp only [e_62_10, e_63_11, e_62_11] at hc
    linear_combination c2428.trans k0 - hc

theorem leafCircuit_f475 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 211)) (a (.virt 295)) (a (.virt 334)) (a (.virt 335)) := by
  have c2429 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2430 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2431 := (leafCircuit_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2432 := (leafCircuit_copies76 a h).1
  have c2433 := (leafCircuit_copies76 a h).2.1
  have c2434 := (leafCircuit_copies76 a h).2.2.1
  have c2435 := (leafCircuit_copies76 a h).2.2.2.1
  have c2436 := (leafCircuit_copies76 a h).2.2.2.2.1
  have c2437 := (leafCircuit_copies76 a h).2.2.2.2.2.1
  have c2438 := (leafCircuit_copies76 a h).2.2.2.2.2.2.1
  have c2439 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.1
  have c2440 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.1
  have c2441 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.1
  have c2442 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2443 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2444 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2445 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_62_13 := arithEq_of_rows h (i := 13) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_13
  simp only [← c2429, k3, ← c2430, k3, ← c2431] at e_62_13
  have e_62_14 := arithEq_of_rows h (i := 14) leafCircuit_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_14
  simp only [← c2432, ← c2433, k3, ← c2434, k6] at e_62_14
  have e_63_12 := arithEq_of_rows h (i := 12) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_12
  simp only [← c2435, ← c2436, ← c2437] at e_63_12
  have e_63_13 := arithEq_of_rows h (i := 13) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_13
  simp only [← c2438, ← c2439, ← c2440] at e_63_13
  have e_65_0 := arithEq_of_rows h (i := 0) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_0
  simp only [← c2441, ← c2442, k3, ← c2443] at e_65_0
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_63_12
    simp only [e_62_14] at hc
    linear_combination c2444.trans k0 - hc
  · have hc := e_65_0
    simp only [e_62_13, e_63_13, e_62_14] at hc
    linear_combination c2445.trans k0 - hc

theorem leafCircuit_f476 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 11) = bselect (a (.virt 328)) (a (.wire 56 31)) (a (.virt 63)) := by
  have c2446 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2447 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2448 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2449 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2450 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2451 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_1 := arithEq_of_rows h (i := 1) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_1
  simp only [← c2446, ← c2447, ← c2448] at e_65_1
  have e_65_2 := arithEq_of_rows h (i := 2) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_2
  simp only [← c2449, ← c2450, ← c2451] at e_65_2
  have hr := e_65_2
  simp only [e_65_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f477 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 19) = bselect (a (.virt 328)) (a (.wire 56 39)) (a (.virt 64)) := by
  have c2452 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2453 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2454 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2455 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2456 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2457 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_3 := arithEq_of_rows h (i := 3) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_3
  simp only [← c2452, ← c2453, ← c2454] at e_65_3
  have e_65_4 := arithEq_of_rows h (i := 4) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_4
  simp only [← c2455, ← c2456, ← c2457] at e_65_4
  have hr := e_65_4
  simp only [e_65_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f478 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 27) = bselect (a (.virt 328)) (a (.wire 56 47)) (a (.virt 65)) := by
  have c2458 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2459 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2460 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2461 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2462 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2463 := (leafCircuit_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_65_5 := arithEq_of_rows h (i := 5) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_5
  simp only [← c2458, ← c2459, ← c2460] at e_65_5
  have e_65_6 := arithEq_of_rows h (i := 6) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_6
  simp only [← c2461, ← c2462, ← c2463] at e_65_6
  have hr := e_65_6
  simp only [e_65_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f479 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 35) = bselect (a (.virt 328)) (a (.wire 56 55)) (a (.virt 66)) := by
  have c2464 := (leafCircuit_copies77 a h).1
  have c2465 := (leafCircuit_copies77 a h).2.1
  have c2466 := (leafCircuit_copies77 a h).2.2.1
  have c2467 := (leafCircuit_copies77 a h).2.2.2.1
  have c2468 := (leafCircuit_copies77 a h).2.2.2.2.1
  have c2469 := (leafCircuit_copies77 a h).2.2.2.2.2.1
  have e_65_7 := arithEq_of_rows h (i := 7) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_7
  simp only [← c2464, ← c2465, ← c2466] at e_65_7
  have e_65_8 := arithEq_of_rows h (i := 8) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_8
  simp only [← c2467, ← c2468, ← c2469] at e_65_8
  have hr := e_65_8
  simp only [e_65_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f480 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 43) = bselect (a (.virt 328)) (a (.virt 63)) (a (.virt 67)) := by
  have c2470 := (leafCircuit_copies77 a h).2.2.2.2.2.2.1
  have c2471 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.1
  have c2472 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.1
  have c2473 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.1
  have c2474 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2475 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_9 := arithEq_of_rows h (i := 9) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_9
  simp only [← c2470, ← c2471, ← c2472] at e_65_9
  have e_65_10 := arithEq_of_rows h (i := 10) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_10
  simp only [← c2473, ← c2474, ← c2475] at e_65_10
  have hr := e_65_10
  simp only [e_65_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f481 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 51) = bselect (a (.virt 330)) (a (.wire 56 31)) (a (.wire 65 43)) := by
  have c2476 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2477 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2478 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2479 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2480 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2481 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_11 := arithEq_of_rows h (i := 11) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_11
  simp only [← c2476, ← c2477, ← c2478] at e_65_11
  have e_65_12 := arithEq_of_rows h (i := 12) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_12
  simp only [← c2479, ← c2480, ← c2481] at e_65_12
  have hr := e_65_12
  simp only [e_65_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f482 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 65 59) = bselect (a (.virt 328)) (a (.virt 64)) (a (.virt 68)) := by
  have c2482 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2483 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2484 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2485 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2486 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2487 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_13 := arithEq_of_rows h (i := 13) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_13
  simp only [← c2482, ← c2483, ← c2484] at e_65_13
  have e_65_14 := arithEq_of_rows h (i := 14) leafCircuit_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_14
  simp only [← c2485, ← c2486, ← c2487] at e_65_14
  have hr := e_65_14
  simp only [e_65_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f483 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 7) = bselect (a (.virt 330)) (a (.wire 56 39)) (a (.wire 65 59)) := by
  have c2488 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2489 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2490 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2491 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2492 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2493 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_66_0 := arithEq_of_rows h (i := 0) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_0
  simp only [← c2488, ← c2489, ← c2490] at e_66_0
  have e_66_1 := arithEq_of_rows h (i := 1) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_1
  simp only [← c2491, ← c2492, ← c2493] at e_66_1
  have hr := e_66_1
  simp only [e_66_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f484 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 15) = bselect (a (.virt 328)) (a (.virt 65)) (a (.virt 69)) := by
  have c2494 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2495 := (leafCircuit_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2496 := (leafCircuit_copies78 a h).1
  have c2497 := (leafCircuit_copies78 a h).2.1
  have c2498 := (leafCircuit_copies78 a h).2.2.1
  have c2499 := (leafCircuit_copies78 a h).2.2.2.1
  have e_66_2 := arithEq_of_rows h (i := 2) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_2
  simp only [← c2494, ← c2495, ← c2496] at e_66_2
  have e_66_3 := arithEq_of_rows h (i := 3) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_3
  simp only [← c2497, ← c2498, ← c2499] at e_66_3
  have hr := e_66_3
  simp only [e_66_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f485 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 23) = bselect (a (.virt 330)) (a (.wire 56 47)) (a (.wire 66 15)) := by
  have c2500 := (leafCircuit_copies78 a h).2.2.2.2.1
  have c2501 := (leafCircuit_copies78 a h).2.2.2.2.2.1
  have c2502 := (leafCircuit_copies78 a h).2.2.2.2.2.2.1
  have c2503 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.1
  have c2504 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.1
  have c2505 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.1
  have e_66_4 := arithEq_of_rows h (i := 4) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_4
  simp only [← c2500, ← c2501, ← c2502] at e_66_4
  have e_66_5 := arithEq_of_rows h (i := 5) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_5
  simp only [← c2503, ← c2504, ← c2505] at e_66_5
  have hr := e_66_5
  simp only [e_66_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f486 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 31) = bselect (a (.virt 328)) (a (.virt 66)) (a (.virt 70)) := by
  have c2506 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2507 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2508 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2509 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2510 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2511 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_66_6 := arithEq_of_rows h (i := 6) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_6
  simp only [← c2506, ← c2507, ← c2508] at e_66_6
  have e_66_7 := arithEq_of_rows h (i := 7) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_7
  simp only [← c2509, ← c2510, ← c2511] at e_66_7
  have hr := e_66_7
  simp only [e_66_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f487 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 39) = bselect (a (.virt 330)) (a (.wire 56 55)) (a (.wire 66 31)) := by
  have c2512 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2513 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2514 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2515 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2516 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2517 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_66_8 := arithEq_of_rows h (i := 8) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_8
  simp only [← c2512, ← c2513, ← c2514] at e_66_8
  have e_66_9 := arithEq_of_rows h (i := 9) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_9
  simp only [← c2515, ← c2516, ← c2517] at e_66_9
  have hr := e_66_9
  simp only [e_66_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f488 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) := by
  have c2518 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2519 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2522 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_7 := arithEq_of_rows h (i := 7) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2518, ← c2519, ← c2520] at e_42_7
  have e_59_5 := arithEq_of_rows h (i := 5) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_5
  simp only [← c2521, ← c2522, k3, ← c2523] at e_59_5
  have hr := e_59_5
  simp only [e_42_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f489 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 47) = bselect (a (.wire 59 23)) (a (.virt 67)) (a (.virt 71)) := by
  have c2524 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2525 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2526 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2527 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2528 := (leafCircuit_copies79 a h).1
  have c2529 := (leafCircuit_copies79 a h).2.1
  have e_66_10 := arithEq_of_rows h (i := 10) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_10
  simp only [← c2524, ← c2525, ← c2526] at e_66_10
  have e_66_11 := arithEq_of_rows h (i := 11) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_11
  simp only [← c2527, ← c2528, ← c2529] at e_66_11
  have hr := e_66_11
  simp only [e_66_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f490 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 66 55) = bselect (a (.virt 332)) (a (.wire 56 31)) (a (.wire 66 47)) := by
  have c2530 := (leafCircuit_copies79 a h).2.2.1
  have c2531 := (leafCircuit_copies79 a h).2.2.2.1
  have c2532 := (leafCircuit_copies79 a h).2.2.2.2.1
  have c2533 := (leafCircuit_copies79 a h).2.2.2.2.2.1
  have c2534 := (leafCircuit_copies79 a h).2.2.2.2.2.2.1
  have c2535 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.1
  have e_66_12 := arithEq_of_rows h (i := 12) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_12
  simp only [← c2530, ← c2531, ← c2532] at e_66_12
  have e_66_13 := arithEq_of_rows h (i := 13) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_13
  simp only [← c2533, ← c2534, ← c2535] at e_66_13
  have hr := e_66_13
  simp only [e_66_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f491 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) := by
  have c2518 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2519 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2522 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_7 := arithEq_of_rows h (i := 7) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2518, ← c2519, ← c2520] at e_42_7
  have e_59_5 := arithEq_of_rows h (i := 5) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_5
  simp only [← c2521, ← c2522, k3, ← c2523] at e_59_5
  have hr := e_59_5
  simp only [e_42_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f492 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 3) = bselect (a (.wire 59 23)) (a (.virt 68)) (a (.virt 72)) := by
  have c2536 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.1
  have c2537 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.1
  have c2538 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2539 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2540 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2541 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_66_14 := arithEq_of_rows h (i := 14) leafCircuit_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_14
  simp only [← c2536, ← c2537, ← c2538] at e_66_14
  have e_67_0 := arithEq_of_rows h (i := 0) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_0
  simp only [← c2539, ← c2540, ← c2541] at e_67_0
  have hr := e_67_0
  simp only [e_66_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f493 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 11) = bselect (a (.virt 332)) (a (.wire 56 39)) (a (.wire 67 3)) := by
  have c2542 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2543 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2544 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2545 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2546 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2547 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_1 := arithEq_of_rows h (i := 1) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_1
  simp only [← c2542, ← c2543, ← c2544] at e_67_1
  have e_67_2 := arithEq_of_rows h (i := 2) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_2
  simp only [← c2545, ← c2546, ← c2547] at e_67_2
  have hr := e_67_2
  simp only [e_67_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f494 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) := by
  have c2518 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2519 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2522 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_7 := arithEq_of_rows h (i := 7) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2518, ← c2519, ← c2520] at e_42_7
  have e_59_5 := arithEq_of_rows h (i := 5) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_5
  simp only [← c2521, ← c2522, k3, ← c2523] at e_59_5
  have hr := e_59_5
  simp only [e_42_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f495 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 19) = bselect (a (.wire 59 23)) (a (.virt 69)) (a (.virt 73)) := by
  have c2548 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2549 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2550 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2551 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2552 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2553 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_3 := arithEq_of_rows h (i := 3) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_3
  simp only [← c2548, ← c2549, ← c2550] at e_67_3
  have e_67_4 := arithEq_of_rows h (i := 4) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_4
  simp only [← c2551, ← c2552, ← c2553] at e_67_4
  have hr := e_67_4
  simp only [e_67_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f496 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 27) = bselect (a (.virt 332)) (a (.wire 56 47)) (a (.wire 67 19)) := by
  have c2554 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2555 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2556 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2557 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2558 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2559 := (leafCircuit_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_67_5 := arithEq_of_rows h (i := 5) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_5
  simp only [← c2554, ← c2555, ← c2556] at e_67_5
  have e_67_6 := arithEq_of_rows h (i := 6) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_6
  simp only [← c2557, ← c2558, ← c2559] at e_67_6
  have hr := e_67_6
  simp only [e_67_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f497 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) := by
  have c2518 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2519 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2522 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (leafCircuit_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_7 := arithEq_of_rows h (i := 7) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2518, ← c2519, ← c2520] at e_42_7
  have e_59_5 := arithEq_of_rows h (i := 5) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_5
  simp only [← c2521, ← c2522, k3, ← c2523] at e_59_5
  have hr := e_59_5
  simp only [e_42_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f498 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 35) = bselect (a (.wire 59 23)) (a (.virt 70)) (a (.virt 74)) := by
  have c2560 := (leafCircuit_copies80 a h).1
  have c2561 := (leafCircuit_copies80 a h).2.1
  have c2562 := (leafCircuit_copies80 a h).2.2.1
  have c2563 := (leafCircuit_copies80 a h).2.2.2.1
  have c2564 := (leafCircuit_copies80 a h).2.2.2.2.1
  have c2565 := (leafCircuit_copies80 a h).2.2.2.2.2.1
  have e_67_7 := arithEq_of_rows h (i := 7) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_7
  simp only [← c2560, ← c2561, ← c2562] at e_67_7
  have e_67_8 := arithEq_of_rows h (i := 8) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_8
  simp only [← c2563, ← c2564, ← c2565] at e_67_8
  have hr := e_67_8
  simp only [e_67_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f499 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 43) = bselect (a (.virt 332)) (a (.wire 56 55)) (a (.wire 67 35)) := by
  have c2566 := (leafCircuit_copies80 a h).2.2.2.2.2.2.1
  have c2567 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.1
  have c2568 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.1
  have c2569 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.1
  have c2570 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2571 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_9 := arithEq_of_rows h (i := 9) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_9
  simp only [← c2566, ← c2567, ← c2568] at e_67_9
  have e_67_10 := arithEq_of_rows h (i := 10) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_10
  simp only [← c2569, ← c2570, ← c2571] at e_67_10
  have hr := e_67_10
  simp only [e_67_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f500 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 51) = bselect (a (.virt 334)) (a (.wire 56 31)) (a (.virt 71)) := by
  have c2572 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2573 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2574 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2575 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2576 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2577 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_11 := arithEq_of_rows h (i := 11) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_11
  simp only [← c2572, ← c2573, ← c2574] at e_67_11
  have e_67_12 := arithEq_of_rows h (i := 12) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_12
  simp only [← c2575, ← c2576, ← c2577] at e_67_12
  have hr := e_67_12
  simp only [e_67_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f501 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 67 59) = bselect (a (.virt 334)) (a (.wire 56 39)) (a (.virt 72)) := by
  have c2578 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2579 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2580 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2581 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2582 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2583 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_13 := arithEq_of_rows h (i := 13) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_13
  simp only [← c2578, ← c2579, ← c2580] at e_67_13
  have e_67_14 := arithEq_of_rows h (i := 14) leafCircuit_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_14
  simp only [← c2581, ← c2582, ← c2583] at e_67_14
  have hr := e_67_14
  simp only [e_67_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f502 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 7) = bselect (a (.virt 334)) (a (.wire 56 47)) (a (.virt 73)) := by
  have c2584 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2585 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2586 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2587 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2588 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2589 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_0 := arithEq_of_rows h (i := 0) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_0
  simp only [← c2584, ← c2585, ← c2586] at e_68_0
  have e_68_1 := arithEq_of_rows h (i := 1) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_1
  simp only [← c2587, ← c2588, ← c2589] at e_68_1
  have hr := e_68_1
  simp only [e_68_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f503 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 15) = bselect (a (.virt 334)) (a (.wire 56 55)) (a (.virt 74)) := by
  have c2590 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2591 := (leafCircuit_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2592 := (leafCircuit_copies81 a h).1
  have c2593 := (leafCircuit_copies81 a h).2.1
  have c2594 := (leafCircuit_copies81 a h).2.2.1
  have c2595 := (leafCircuit_copies81 a h).2.2.2.1
  have e_68_2 := arithEq_of_rows h (i := 2) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_2
  simp only [← c2590, ← c2591, ← c2592] at e_68_2
  have e_68_3 := arithEq_of_rows h (i := 3) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_3
  simp only [← c2593, ← c2594, ← c2595] at e_68_3
  have hr := e_68_3
  simp only [e_68_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f504 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 71 12) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 0 ∧ a (.wire 71 13) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 1 ∧ a (.wire 71 14) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 2 ∧ a (.wire 71 15) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 3) := by
  have c2596 := (leafCircuit_copies81 a h).2.2.2.2.1
  have c2597 := (leafCircuit_copies81 a h).2.2.2.2.2.1
  have c2598 := (leafCircuit_copies81 a h).2.2.2.2.2.2.1
  have c2599 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.1
  have c2600 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.1
  have c2601 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.1
  have c2602 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2603 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2604 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2605 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2606 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2607 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2608 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2609 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2610 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2611 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2612 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2613 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2614 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2615 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2616 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2617 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2618 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2619 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2620 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2621 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2622 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2623 := (leafCircuit_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2624 := (leafCircuit_copies82 a h).1
  have c2625 := (leafCircuit_copies82 a h).2.1
  have c2626 := (leafCircuit_copies82 a h).2.2.1
  have c2627 := (leafCircuit_copies82 a h).2.2.2.1
  have c2628 := (leafCircuit_copies82 a h).2.2.2.2.1
  have c2629 := (leafCircuit_copies82 a h).2.2.2.2.2.1
  have c2630 := (leafCircuit_copies82 a h).2.2.2.2.2.2.1
  have c2631 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.1
  have c2632 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.1
  have c2633 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.1
  have c2634 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2635 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2636 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2637 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2638 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2639 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2640 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2641 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2642 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2643 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2644 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2645 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2646 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2647 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2648 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2649 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2650 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2651 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2652 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2653 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2654 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2655 := (leafCircuit_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2656 := (leafCircuit_copies83 a h).1
  have c2657 := (leafCircuit_copies83 a h).2.1
  have c2658 := (leafCircuit_copies83 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_59_6 := arithEq_of_rows h (i := 6) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_6
  simp only [← c2608, ← c2609, k3, ← c2610] at e_59_6
  have e_59_7 := arithEq_of_rows h (i := 7) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_7
  simp only [← c2611, ← c2612, k3, ← c2613] at e_59_7
  have e_59_8 := arithEq_of_rows h (i := 8) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_8
  simp only [← c2614, ← c2615, k3, ← c2616] at e_59_8
  have e_59_9 := arithEq_of_rows h (i := 9) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_9
  simp only [← c2617, ← c2618, k3, ← c2619] at e_59_9
  have e_59_10 := arithEq_of_rows h (i := 10) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_10
  simp only [← c2620, ← c2621, k3, ← c2622] at e_59_10
  have e_59_11 := arithEq_of_rows h (i := 11) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_11
  simp only [← c2623, ← c2624, k3, ← c2625] at e_59_11
  have e_59_12 := arithEq_of_rows h (i := 12) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_12
  simp only [← c2626, ← c2627, k3, ← c2628] at e_59_12
  have e_59_13 := arithEq_of_rows h (i := 13) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_13
  simp only [← c2629, ← c2630, k3, ← c2631] at e_59_13
  have e_59_14 := arithEq_of_rows h (i := 14) leafCircuit_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_14
  simp only [← c2644, ← c2645, k3, ← c2646, k3] at e_59_14
  have hin0 : poseidon2In a 69 = addBlock (fun _ => 0) [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39)] :=
    poseidon2In_first c2596.symm c2597.symm c2598.symm c2599.symm c2600.symm c2601.symm c2602.symm c2603.symm (c2604.symm.trans k0) (c2605.symm.trans k0) (c2606.symm.trans k0) (c2607.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 69) rfl rfl hin0
  have hin1 : poseidon2In a 70 = addBlock (poseidon2Out a 69) [a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] :=
    poseidon2In_chain (by linear_combination -c2632 + e_59_6) (by linear_combination -c2633 + e_59_7) (by linear_combination -c2634 + e_59_8) (by linear_combination -c2635 + e_59_9) (by linear_combination -c2636 + e_59_10) (by linear_combination -c2637 + e_59_11) (by linear_combination -c2638 + e_59_12) (by linear_combination -c2639 + e_59_13) c2640.symm c2641.symm c2642.symm c2643.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 70) rfl rfl hin1
  have hin2 : poseidon2In a 71 = addBlock (poseidon2Out a 70) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c2647 + e_59_14) (c2648.symm.trans (add_zero _).symm) (c2649.symm.trans (add_zero _).symm) (c2650.symm.trans (add_zero _).symm) (c2651.symm.trans (add_zero _).symm) (c2652.symm.trans (add_zero _).symm) (c2653.symm.trans (add_zero _).symm) (c2654.symm.trans (add_zero _).symm) c2655.symm c2656.symm c2657.symm c2658.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 71) rfl rfl hin2
  have hpad : pad10 [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] =
      [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39)] ++ ([a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f505 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 23) = bselect (a (.wire 59 19)) (a (.wire 71 12)) (a (.wire 56 31)) := by
  have c2659 := (leafCircuit_copies83 a h).2.2.2.1
  have c2660 := (leafCircuit_copies83 a h).2.2.2.2.1
  have c2661 := (leafCircuit_copies83 a h).2.2.2.2.2.1
  have c2662 := (leafCircuit_copies83 a h).2.2.2.2.2.2.1
  have c2663 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.1
  have c2664 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.1
  have e_68_4 := arithEq_of_rows h (i := 4) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_4
  simp only [← c2659, ← c2660, ← c2661] at e_68_4
  have e_68_5 := arithEq_of_rows h (i := 5) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_5
  simp only [← c2662, ← c2663, ← c2664] at e_68_5
  have hr := e_68_5
  simp only [e_68_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f506 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 31) = bselect (a (.wire 59 19)) (a (.wire 71 13)) (a (.wire 56 39)) := by
  have c2665 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.1
  have c2666 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2667 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2668 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2669 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2670 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_6 := arithEq_of_rows h (i := 6) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_6
  simp only [← c2665, ← c2666, ← c2667] at e_68_6
  have e_68_7 := arithEq_of_rows h (i := 7) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_7
  simp only [← c2668, ← c2669, ← c2670] at e_68_7
  have hr := e_68_7
  simp only [e_68_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f507 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 39) = bselect (a (.wire 59 19)) (a (.wire 71 14)) (a (.wire 56 47)) := by
  have c2671 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2672 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2673 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2674 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2675 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2676 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_8 := arithEq_of_rows h (i := 8) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_8
  simp only [← c2671, ← c2672, ← c2673] at e_68_8
  have e_68_9 := arithEq_of_rows h (i := 9) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_9
  simp only [← c2674, ← c2675, ← c2676] at e_68_9
  have hr := e_68_9
  simp only [e_68_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f508 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 47) = bselect (a (.wire 59 19)) (a (.wire 71 15)) (a (.wire 56 55)) := by
  have c2677 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2678 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2679 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2680 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2681 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2682 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_10 := arithEq_of_rows h (i := 10) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_10
  simp only [← c2677, ← c2678, ← c2679] at e_68_10
  have e_68_11 := arithEq_of_rows h (i := 11) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_11
  simp only [← c2680, ← c2681, ← c2682] at e_68_11
  have hr := e_68_11
  simp only [e_68_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f509 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 72 1), a (.wire 72 2), a (.wire 72 3), a (.wire 72 4), a (.wire 72 5)] := by
  have c2683 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2684 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2685 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2686 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2687 := (leafCircuit_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2688 := (leafCircuit_copies84 a h).1
  have c2689 := (leafCircuit_copies84 a h).2.1
  have c2690 := (leafCircuit_copies84 a h).2.2.1
  have c2691 := (leafCircuit_copies84 a h).2.2.2.1
  have c2692 := (leafCircuit_copies84 a h).2.2.2.2.1
  have c2693 := (leafCircuit_copies84 a h).2.2.2.2.2.1
  have c2694 := (leafCircuit_copies84 a h).2.2.2.2.2.2.1
  have c2695 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.1
  have c2696 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.1
  have c2697 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.1
  have c2698 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2699 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2700 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2701 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2702 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2703 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2704 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2705 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2706 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2707 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2708 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2709 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2710 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2711 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2712 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2713 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2714 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2715 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2716 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2717 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2718 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2719 := (leafCircuit_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2720 := (leafCircuit_copies85 a h).1
  have c2721 := (leafCircuit_copies85 a h).2.1
  have c2722 := (leafCircuit_copies85 a h).2.2.1
  have c2723 := (leafCircuit_copies85 a h).2.2.2.1
  have c2724 := (leafCircuit_copies85 a h).2.2.2.2.1
  have c2725 := (leafCircuit_copies85 a h).2.2.2.2.2.1
  have c2726 := (leafCircuit_copies85 a h).2.2.2.2.2.2.1
  have c2727 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.1
  have c2728 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.1
  have c2729 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.1
  have c2730 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2731 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2732 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2733 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2734 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2735 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2736 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2737 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 72) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2683.trans k0
    · exact c2684.trans k0
    · exact c2685.trans k0
    · exact c2686.trans k0
    · exact c2687.trans k0
    · exact c2688.trans k0
    · exact c2689.trans k0
    · exact c2690.trans k0
    · exact c2691.trans k0
    · exact c2692.trans k0
    · exact c2693.trans k0
    · exact c2694.trans k0
    · exact c2695.trans k0
    · exact c2696.trans k0
    · exact c2697.trans k0
    · exact c2698.trans k0
    · exact c2699.trans k0
    · exact c2700.trans k0
    · exact c2701.trans k0
    · exact c2702.trans k0
    · exact c2703.trans k0
    · exact c2704.trans k0
    · exact c2705.trans k0
    · exact c2706.trans k0
    · exact c2707.trans k0
    · exact c2708.trans k0
    · exact c2709.trans k0
    · exact c2710.trans k0
    · exact c2711.trans k0
    · exact c2712.trans k0
    · exact c2713.trans k0
    · exact c2714.trans k0
    · exact c2715.trans k0
    · exact c2716.trans k0
    · exact c2717.trans k0
    · exact c2718.trans k0
    · exact c2719.trans k0
    · exact c2720.trans k0
    · exact c2721.trans k0
    · exact c2722.trans k0
    · exact c2723.trans k0
    · exact c2724.trans k0
    · exact c2725.trans k0
    · exact c2726.trans k0
    · exact c2727.trans k0
    · exact c2728.trans k0
    · exact c2729.trans k0
    · exact c2730.trans k0
    · exact c2731.trans k0
    · exact c2732.trans k0
    · exact c2733.trans k0
    · exact c2734.trans k0
    · exact c2735.trans k0
    · exact c2736.trans k0
    )
  rwa [c2737] at hr

theorem leafCircuit_f510 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f511 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 5) = band (a (.virt 292)) (a (.wire 72 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f512 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 5) = band (a (.wire 72 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f513 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 5) = bor (a (.virt 14)) (a (.wire 72 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f514 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 72 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f515 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f516 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 5) = a (.virt 14) + a (.wire 72 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f517 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 5) = a (.wire 72 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f518 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 51) = bnot (a (.wire 72 5)) := by
  have c2738 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2739 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2740 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_68_12 := arithEq_of_rows h (i := 12) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_12
  simp only [← c2738, k3, ← c2739, k3, ← c2740] at e_68_12
  have hr := e_68_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f519 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 51) = band (a (.virt 292)) (a (.wire 68 51)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f520 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f521 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 4) = band (a (.virt 292)) (a (.wire 72 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f522 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 63 59) = band (a (.wire 72 4)) (a (.wire 68 51)) := by
  have c2741 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2742 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2743 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_14 := arithEq_of_rows h (i := 14) leafCircuit_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_14
  simp only [← c2741, ← c2742, ← c2743] at e_63_14
  have hr := e_63_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f523 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 3) = bor (a (.wire 72 5)) (a (.wire 63 59)) := by
  have c2744 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2745 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2746 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2747 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2748 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2749 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_8 := arithEq_of_rows h (i := 8) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_8
  simp only [← c2744, ← c2745, ← c2746] at e_42_8
  have e_73_0 := arithEq_of_rows h (i := 0) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_0
  simp only [← c2747, ← c2748, k3, ← c2749] at e_73_0
  have hr := e_73_0
  simp only [e_42_8] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f524 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 72 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f525 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f526 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 4) = a (.virt 14) + a (.wire 72 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f527 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 4) = a (.wire 72 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f528 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 55) = bnot (a (.wire 72 4)) := by
  have c2750 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2751 := (leafCircuit_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2752 := (leafCircuit_copies86 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_68_13 := arithEq_of_rows h (i := 13) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_13
  simp only [← c2750, k3, ← c2751, k3, ← c2752] at e_68_13
  have hr := e_68_13
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f529 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 3) = band (a (.wire 68 51)) (a (.wire 68 55)) := by
  have c2753 := (leafCircuit_copies86 a h).2.1
  have c2754 := (leafCircuit_copies86 a h).2.2.1
  have c2755 := (leafCircuit_copies86 a h).2.2.2.1
  have e_74_0 := arithEq_of_rows h (i := 0) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_0
  simp only [← c2753, ← c2754, ← c2755] at e_74_0
  have hr := e_74_0
  simp only [band]
  linear_combination hr

theorem leafCircuit_f530 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f531 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 72 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f532 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 74 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f533 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 3) = bor (a (.wire 73 3)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f534 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 3) = a (.virt 292) * a (.wire 72 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f535 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 7) = a (.virt 294) * a (.wire 72 3) := by
  have c2756 := (leafCircuit_copies86 a h).2.2.2.2.1
  have c2757 := (leafCircuit_copies86 a h).2.2.2.2.2.1
  have c2758 := (leafCircuit_copies86 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_74_1 := arithEq_of_rows h (i := 1) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_1
  simp only [← c2756, k5, ← c2757, ← c2758, k5] at e_74_1
  have hr := e_74_1
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f536 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 7) = a (.virt 292) + a (.wire 72 3) := by
  have c2759 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.1
  have c2760 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.1
  have c2761 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_73_1 := arithEq_of_rows h (i := 1) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_1
  simp only [← c2759, k3, ← c2760, k3, ← c2761] at e_73_1
  have hr := e_73_1
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f537 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 68 59) = a (.wire 73 7) - a (.wire 74 7) := by
  have c2762 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2763 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2764 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_68_14 := arithEq_of_rows h (i := 14) leafCircuit_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_14
  simp only [← c2762, ← c2763, k3, ← c2764] at e_68_14
  have hr := e_68_14
  linear_combination hr

theorem leafCircuit_f538 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 75 3) = bnot (a (.wire 68 59)) := by
  have c2765 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2766 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2767 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_75_0 := arithEq_of_rows h (i := 0) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_0
  simp only [← c2765, k3, ← c2766, k3, ← c2767] at e_75_0
  have hr := e_75_0
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f539 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 11) = band (a (.wire 74 3)) (a (.wire 75 3)) := by
  have c2768 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2769 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2770 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_2 := arithEq_of_rows h (i := 2) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_2
  simp only [← c2768, ← c2769, ← c2770] at e_74_2
  have hr := e_74_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f540 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f541 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 2) = band (a (.virt 292)) (a (.wire 72 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f542 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 15) = band (a (.wire 72 2)) (a (.wire 74 11)) := by
  have c2771 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2772 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2773 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_3 := arithEq_of_rows h (i := 3) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_3
  simp only [← c2771, ← c2772, ← c2773] at e_74_3
  have hr := e_74_3
  simp only [band]
  linear_combination hr

theorem leafCircuit_f543 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 11) = bor (a (.wire 73 3)) (a (.wire 74 15)) := by
  have c2774 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2775 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2776 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2777 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2778 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2779 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_9 := arithEq_of_rows h (i := 9) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_9
  simp only [← c2774, ← c2775, ← c2776] at e_42_9
  have e_73_2 := arithEq_of_rows h (i := 2) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_2
  simp only [← c2777, ← c2778, k3, ← c2779] at e_73_2
  have hr := e_73_2
  simp only [e_42_9] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f544 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 72 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f545 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f546 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 2) = a (.virt 14) + a (.wire 72 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f547 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 2) = a (.wire 72 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f548 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 75 7) = bnot (a (.wire 72 2)) := by
  have c2780 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2781 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2782 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_75_1 := arithEq_of_rows h (i := 1) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_1
  simp only [← c2780, k3, ← c2781, k3, ← c2782] at e_75_1
  have hr := e_75_1
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f549 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 19) = band (a (.wire 74 11)) (a (.wire 75 7)) := by
  have c2783 := (leafCircuit_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2784 := (leafCircuit_copies87 a h).1
  have c2785 := (leafCircuit_copies87 a h).2.1
  have e_74_4 := arithEq_of_rows h (i := 4) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_4
  simp only [← c2783, ← c2784, ← c2785] at e_74_4
  have hr := e_74_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f550 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f551 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 72 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f552 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 74 19)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f553 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 11) = bor (a (.wire 73 11)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f554 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 72 1) = a (.virt 292) * a (.wire 72 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f555 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 23) = a (.virt 294) * a (.wire 72 1) := by
  have c2786 := (leafCircuit_copies87 a h).2.2.1
  have c2787 := (leafCircuit_copies87 a h).2.2.2.1
  have c2788 := (leafCircuit_copies87 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_74_5 := arithEq_of_rows h (i := 5) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_5
  simp only [← c2786, k5, ← c2787, ← c2788, k5] at e_74_5
  have hr := e_74_5
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f556 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 15) = a (.virt 292) + a (.wire 72 1) := by
  have c2789 := (leafCircuit_copies87 a h).2.2.2.2.2.1
  have c2790 := (leafCircuit_copies87 a h).2.2.2.2.2.2.1
  have c2791 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_73_3 := arithEq_of_rows h (i := 3) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_3
  simp only [← c2789, k3, ← c2790, k3, ← c2791] at e_73_3
  have hr := e_73_3
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f557 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 75 11) = a (.wire 73 15) - a (.wire 74 23) := by
  have c2792 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.1
  have c2793 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.1
  have c2794 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_75_2 := arithEq_of_rows h (i := 2) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_2
  simp only [← c2792, ← c2793, k3, ← c2794] at e_75_2
  have hr := e_75_2
  linear_combination hr

theorem leafCircuit_f558 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 75 15) = bnot (a (.wire 75 11)) := by
  have c2795 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2796 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2797 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_75_3 := arithEq_of_rows h (i := 3) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_3
  simp only [← c2795, k3, ← c2796, k3, ← c2797] at e_75_3
  have hr := e_75_3
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f559 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 74 27) = band (a (.wire 74 19)) (a (.wire 75 15)) := by
  have c2798 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2799 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2800 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_74_6 := arithEq_of_rows h (i := 6) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_6
  simp only [← c2798, ← c2799, ← c2800] at e_74_6
  have hr := e_74_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f560 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 212)) 2 := by
  have c2801 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2802 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2803 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2804 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2805 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2806 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2807 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2808 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2809 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2810 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2811 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2812 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2813 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2814 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2815 := (leafCircuit_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2816 := (leafCircuit_copies88 a h).1
  have c2817 := (leafCircuit_copies88 a h).2.1
  have c2818 := (leafCircuit_copies88 a h).2.2.1
  have c2819 := (leafCircuit_copies88 a h).2.2.2.1
  have c2820 := (leafCircuit_copies88 a h).2.2.2.2.1
  have c2821 := (leafCircuit_copies88 a h).2.2.2.2.2.1
  have c2822 := (leafCircuit_copies88 a h).2.2.2.2.2.2.1
  have c2823 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.1
  have c2824 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.1
  have c2825 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.1
  have c2826 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2827 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2828 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2829 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2830 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2831 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2832 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2833 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2834 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2835 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2836 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2837 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2838 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2839 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2840 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2841 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2842 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2843 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2844 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2845 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2846 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2847 := (leafCircuit_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2848 := (leafCircuit_copies89 a h).1
  have c2849 := (leafCircuit_copies89 a h).2.1
  have c2850 := (leafCircuit_copies89 a h).2.2.1
  have c2851 := (leafCircuit_copies89 a h).2.2.2.1
  have c2852 := (leafCircuit_copies89 a h).2.2.2.2.1
  have c2853 := (leafCircuit_copies89 a h).2.2.2.2.2.1
  have c2854 := (leafCircuit_copies89 a h).2.2.2.2.2.2.1
  have c2855 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.1
  have c2856 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.1
  have c2857 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.1
  have c2858 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 76) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2801.trans k0
    · exact c2802.trans k0
    · exact c2803.trans k0
    · exact c2804.trans k0
    · exact c2805.trans k0
    · exact c2806.trans k0
    · exact c2807.trans k0
    · exact c2808.trans k0
    · exact c2809.trans k0
    · exact c2810.trans k0
    · exact c2811.trans k0
    · exact c2812.trans k0
    · exact c2813.trans k0
    · exact c2814.trans k0
    · exact c2815.trans k0
    · exact c2816.trans k0
    · exact c2817.trans k0
    · exact c2818.trans k0
    · exact c2819.trans k0
    · exact c2820.trans k0
    · exact c2821.trans k0
    · exact c2822.trans k0
    · exact c2823.trans k0
    · exact c2824.trans k0
    · exact c2825.trans k0
    · exact c2826.trans k0
    · exact c2827.trans k0
    · exact c2828.trans k0
    · exact c2829.trans k0
    · exact c2830.trans k0
    · exact c2831.trans k0
    · exact c2832.trans k0
    · exact c2833.trans k0
    · exact c2834.trans k0
    · exact c2835.trans k0
    · exact c2836.trans k0
    · exact c2837.trans k0
    · exact c2838.trans k0
    · exact c2839.trans k0
    · exact c2840.trans k0
    · exact c2841.trans k0
    · exact c2842.trans k0
    · exact c2843.trans k0
    · exact c2844.trans k0
    · exact c2845.trans k0
    · exact c2846.trans k0
    · exact c2847.trans k0
    · exact c2848.trans k0
    · exact c2849.trans k0
    · exact c2850.trans k0
    · exact c2851.trans k0
    · exact c2852.trans k0
    · exact c2853.trans k0
    · exact c2854.trans k0
    · exact c2855.trans k0
    · exact c2856.trans k0
    · exact c2857.trans k0
    )
  rwa [c2858] at hr

theorem leafCircuit_f561 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 212)) (a (.virt 14)) (a (.virt 336)) (a (.virt 337)) := by
  have c2859 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2860 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2861 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2862 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2863 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2864 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2865 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2866 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2867 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2868 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2869 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2870 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2871 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2872 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_74_7 := arithEq_of_rows h (i := 7) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_7
  simp only [← c2862, ← c2863, ← c2864] at e_74_7
  have e_74_8 := arithEq_of_rows h (i := 8) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_8
  simp only [← c2865, ← c2866, ← c2867] at e_74_8
  have e_75_4 := arithEq_of_rows h (i := 4) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_4
  simp only [← c2859, k3, ← c2860, k3, ← c2861] at e_75_4
  have e_75_5 := arithEq_of_rows h (i := 5) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_5
  simp only [← c2868, ← c2869, k3, ← c2870] at e_75_5
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_74_7
    linear_combination c2871.trans k0 - hc
  · have hc := e_75_5
    simp only [e_75_4, e_74_8] at hc
    linear_combination c2872.trans k0 - hc

theorem leafCircuit_f562 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 212)) (a (.virt 292)) (a (.virt 338)) (a (.virt 339)) := by
  have c2873 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2874 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2875 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2876 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2877 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2878 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2879 := (leafCircuit_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2880 := (leafCircuit_copies90 a h).1
  have c2881 := (leafCircuit_copies90 a h).2.1
  have c2882 := (leafCircuit_copies90 a h).2.2.1
  have c2883 := (leafCircuit_copies90 a h).2.2.2.1
  have c2884 := (leafCircuit_copies90 a h).2.2.2.2.1
  have c2885 := (leafCircuit_copies90 a h).2.2.2.2.2.1
  have c2886 := (leafCircuit_copies90 a h).2.2.2.2.2.2.1
  have c2887 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.1
  have c2888 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.1
  have c2889 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_74_9 := arithEq_of_rows h (i := 9) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_9
  simp only [← c2879, ← c2880, ← c2881] at e_74_9
  have e_74_10 := arithEq_of_rows h (i := 10) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_10
  simp only [← c2882, ← c2883, ← c2884] at e_74_10
  have e_75_6 := arithEq_of_rows h (i := 6) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_6
  simp only [← c2873, k3, ← c2874, k3, ← c2875] at e_75_6
  have e_75_7 := arithEq_of_rows h (i := 7) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_7
  simp only [← c2876, ← c2877, k3, ← c2878, k3] at e_75_7
  have e_75_8 := arithEq_of_rows h (i := 8) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_8
  simp only [← c2885, ← c2886, k3, ← c2887] at e_75_8
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_74_9
    simp only [e_75_7] at hc
    linear_combination c2888.trans k0 - hc
  · have hc := e_75_8
    simp only [e_75_6, e_74_10, e_75_7] at hc
    linear_combination c2889.trans k0 - hc

theorem leafCircuit_f563 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 212)) (a (.virt 294)) (a (.virt 340)) (a (.virt 341)) := by
  have c2890 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2891 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2892 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2893 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2894 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2895 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2896 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2897 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2898 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2899 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2900 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2901 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2902 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2903 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2904 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2905 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2906 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_74_11 := arithEq_of_rows h (i := 11) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_11
  simp only [← c2896, ← c2897, ← c2898] at e_74_11
  have e_74_12 := arithEq_of_rows h (i := 12) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_12
  simp only [← c2899, ← c2900, ← c2901] at e_74_12
  have e_75_9 := arithEq_of_rows h (i := 9) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_9
  simp only [← c2890, k3, ← c2891, k3, ← c2892] at e_75_9
  have e_75_10 := arithEq_of_rows h (i := 10) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_10
  simp only [← c2893, ← c2894, k3, ← c2895, k5] at e_75_10
  have e_75_11 := arithEq_of_rows h (i := 11) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_11
  simp only [← c2902, ← c2903, k3, ← c2904] at e_75_11
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_74_11
    simp only [e_75_10] at hc
    linear_combination c2905.trans k0 - hc
  · have hc := e_75_11
    simp only [e_75_9, e_74_12, e_75_10] at hc
    linear_combination c2906.trans k0 - hc

theorem leafCircuit_f564 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 212)) (a (.virt 295)) (a (.virt 342)) (a (.virt 343)) := by
  have c2907 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2908 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2909 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2910 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2911 := (leafCircuit_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2912 := (leafCircuit_copies91 a h).1
  have c2913 := (leafCircuit_copies91 a h).2.1
  have c2914 := (leafCircuit_copies91 a h).2.2.1
  have c2915 := (leafCircuit_copies91 a h).2.2.2.1
  have c2916 := (leafCircuit_copies91 a h).2.2.2.2.1
  have c2917 := (leafCircuit_copies91 a h).2.2.2.2.2.1
  have c2918 := (leafCircuit_copies91 a h).2.2.2.2.2.2.1
  have c2919 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.1
  have c2920 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.1
  have c2921 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.1
  have c2922 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2923 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_74_13 := arithEq_of_rows h (i := 13) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_13
  simp only [← c2913, ← c2914, ← c2915] at e_74_13
  have e_74_14 := arithEq_of_rows h (i := 14) leafCircuit_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_14
  simp only [← c2916, ← c2917, ← c2918] at e_74_14
  have e_75_12 := arithEq_of_rows h (i := 12) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_12
  simp only [← c2907, k3, ← c2908, k3, ← c2909] at e_75_12
  have e_75_13 := arithEq_of_rows h (i := 13) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_13
  simp only [← c2910, ← c2911, k3, ← c2912, k6] at e_75_13
  have e_75_14 := arithEq_of_rows h (i := 14) leafCircuit_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_14
  simp only [← c2919, ← c2920, k3, ← c2921] at e_75_14
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_74_13
    simp only [e_75_13] at hc
    linear_combination c2922.trans k0 - hc
  · have hc := e_75_14
    simp only [e_75_12, e_74_14, e_75_13] at hc
    linear_combination c2923.trans k0 - hc

theorem leafCircuit_f565 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 7) = bselect (a (.virt 336)) (a (.wire 68 23)) (a (.virt 75)) := by
  have c2924 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2925 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2926 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2927 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2928 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2929 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_0 := arithEq_of_rows h (i := 0) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_0
  simp only [← c2924, ← c2925, ← c2926] at e_77_0
  have e_77_1 := arithEq_of_rows h (i := 1) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_1
  simp only [← c2927, ← c2928, ← c2929] at e_77_1
  have hr := e_77_1
  simp only [e_77_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f566 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 15) = bselect (a (.virt 336)) (a (.wire 68 31)) (a (.virt 76)) := by
  have c2930 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2931 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2932 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2933 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2934 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2935 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_2 := arithEq_of_rows h (i := 2) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_2
  simp only [← c2930, ← c2931, ← c2932] at e_77_2
  have e_77_3 := arithEq_of_rows h (i := 3) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_3
  simp only [← c2933, ← c2934, ← c2935] at e_77_3
  have hr := e_77_3
  simp only [e_77_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f567 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 23) = bselect (a (.virt 336)) (a (.wire 68 39)) (a (.virt 77)) := by
  have c2936 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2937 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2938 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2939 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2940 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2941 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_4 := arithEq_of_rows h (i := 4) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_4
  simp only [← c2936, ← c2937, ← c2938] at e_77_4
  have e_77_5 := arithEq_of_rows h (i := 5) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_5
  simp only [← c2939, ← c2940, ← c2941] at e_77_5
  have hr := e_77_5
  simp only [e_77_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f568 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 31) = bselect (a (.virt 336)) (a (.wire 68 47)) (a (.virt 78)) := by
  have c2942 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2943 := (leafCircuit_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2944 := (leafCircuit_copies92 a h).1
  have c2945 := (leafCircuit_copies92 a h).2.1
  have c2946 := (leafCircuit_copies92 a h).2.2.1
  have c2947 := (leafCircuit_copies92 a h).2.2.2.1
  have e_77_6 := arithEq_of_rows h (i := 6) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_6
  simp only [← c2942, ← c2943, ← c2944] at e_77_6
  have e_77_7 := arithEq_of_rows h (i := 7) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_7
  simp only [← c2945, ← c2946, ← c2947] at e_77_7
  have hr := e_77_7
  simp only [e_77_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f569 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 39) = bselect (a (.virt 336)) (a (.virt 75)) (a (.virt 79)) := by
  have c2948 := (leafCircuit_copies92 a h).2.2.2.2.1
  have c2949 := (leafCircuit_copies92 a h).2.2.2.2.2.1
  have c2950 := (leafCircuit_copies92 a h).2.2.2.2.2.2.1
  have c2951 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.1
  have c2952 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.1
  have c2953 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.1
  have e_77_8 := arithEq_of_rows h (i := 8) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_8
  simp only [← c2948, ← c2949, ← c2950] at e_77_8
  have e_77_9 := arithEq_of_rows h (i := 9) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_9
  simp only [← c2951, ← c2952, ← c2953] at e_77_9
  have hr := e_77_9
  simp only [e_77_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f570 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 47) = bselect (a (.virt 338)) (a (.wire 68 23)) (a (.wire 77 39)) := by
  have c2954 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2955 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2956 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2957 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2958 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2959 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_10 := arithEq_of_rows h (i := 10) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_10
  simp only [← c2954, ← c2955, ← c2956] at e_77_10
  have e_77_11 := arithEq_of_rows h (i := 11) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_11
  simp only [← c2957, ← c2958, ← c2959] at e_77_11
  have hr := e_77_11
  simp only [e_77_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f571 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 77 55) = bselect (a (.virt 336)) (a (.virt 76)) (a (.virt 80)) := by
  have c2960 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2961 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2962 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2963 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2964 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2965 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_12 := arithEq_of_rows h (i := 12) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_12
  simp only [← c2960, ← c2961, ← c2962] at e_77_12
  have e_77_13 := arithEq_of_rows h (i := 13) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_13
  simp only [← c2963, ← c2964, ← c2965] at e_77_13
  have hr := e_77_13
  simp only [e_77_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f572 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 3) = bselect (a (.virt 338)) (a (.wire 68 31)) (a (.wire 77 55)) := by
  have c2966 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2967 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2968 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2969 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2970 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2971 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_14 := arithEq_of_rows h (i := 14) leafCircuit_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_14
  simp only [← c2966, ← c2967, ← c2968] at e_77_14
  have e_78_0 := arithEq_of_rows h (i := 0) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_0
  simp only [← c2969, ← c2970, ← c2971] at e_78_0
  have hr := e_78_0
  simp only [e_77_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f573 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 11) = bselect (a (.virt 336)) (a (.virt 77)) (a (.virt 81)) := by
  have c2972 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2973 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2974 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2975 := (leafCircuit_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2976 := (leafCircuit_copies93 a h).1
  have c2977 := (leafCircuit_copies93 a h).2.1
  have e_78_1 := arithEq_of_rows h (i := 1) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_1
  simp only [← c2972, ← c2973, ← c2974] at e_78_1
  have e_78_2 := arithEq_of_rows h (i := 2) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_2
  simp only [← c2975, ← c2976, ← c2977] at e_78_2
  have hr := e_78_2
  simp only [e_78_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f574 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 19) = bselect (a (.virt 338)) (a (.wire 68 39)) (a (.wire 78 11)) := by
  have c2978 := (leafCircuit_copies93 a h).2.2.1
  have c2979 := (leafCircuit_copies93 a h).2.2.2.1
  have c2980 := (leafCircuit_copies93 a h).2.2.2.2.1
  have c2981 := (leafCircuit_copies93 a h).2.2.2.2.2.1
  have c2982 := (leafCircuit_copies93 a h).2.2.2.2.2.2.1
  have c2983 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.1
  have e_78_3 := arithEq_of_rows h (i := 3) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_3
  simp only [← c2978, ← c2979, ← c2980] at e_78_3
  have e_78_4 := arithEq_of_rows h (i := 4) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_4
  simp only [← c2981, ← c2982, ← c2983] at e_78_4
  have hr := e_78_4
  simp only [e_78_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f575 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 27) = bselect (a (.virt 336)) (a (.virt 78)) (a (.virt 82)) := by
  have c2984 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.1
  have c2985 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.1
  have c2986 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2987 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2988 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2989 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_5 := arithEq_of_rows h (i := 5) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_5
  simp only [← c2984, ← c2985, ← c2986] at e_78_5
  have e_78_6 := arithEq_of_rows h (i := 6) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_6
  simp only [← c2987, ← c2988, ← c2989] at e_78_6
  have hr := e_78_6
  simp only [e_78_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f576 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 35) = bselect (a (.virt 338)) (a (.wire 68 47)) (a (.wire 78 27)) := by
  have c2990 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2991 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2992 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2993 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2994 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2995 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_7 := arithEq_of_rows h (i := 7) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_7
  simp only [← c2990, ← c2991, ← c2992] at e_78_7
  have e_78_8 := arithEq_of_rows h (i := 8) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_8
  simp only [← c2993, ← c2994, ← c2995] at e_78_8
  have hr := e_78_8
  simp only [e_78_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f577 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) := by
  have c2996 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_10 := arithEq_of_rows h (i := 10) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2996, ← c2997, ← c2998] at e_42_10
  have e_73_4 := arithEq_of_rows h (i := 4) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c2999, ← c3000, k3, ← c3001] at e_73_4
  have hr := e_73_4
  simp only [e_42_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f578 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 43) = bselect (a (.wire 73 19)) (a (.virt 79)) (a (.virt 83)) := by
  have c3002 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3003 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3004 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3005 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3006 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3007 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_78_9 := arithEq_of_rows h (i := 9) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_9
  simp only [← c3002, ← c3003, ← c3004] at e_78_9
  have e_78_10 := arithEq_of_rows h (i := 10) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_10
  simp only [← c3005, ← c3006, ← c3007] at e_78_10
  have hr := e_78_10
  simp only [e_78_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f579 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 51) = bselect (a (.virt 340)) (a (.wire 68 23)) (a (.wire 78 43)) := by
  have c3008 := (leafCircuit_copies94 a h).1
  have c3009 := (leafCircuit_copies94 a h).2.1
  have c3010 := (leafCircuit_copies94 a h).2.2.1
  have c3011 := (leafCircuit_copies94 a h).2.2.2.1
  have c3012 := (leafCircuit_copies94 a h).2.2.2.2.1
  have c3013 := (leafCircuit_copies94 a h).2.2.2.2.2.1
  have e_78_11 := arithEq_of_rows h (i := 11) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_11
  simp only [← c3008, ← c3009, ← c3010] at e_78_11
  have e_78_12 := arithEq_of_rows h (i := 12) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_12
  simp only [← c3011, ← c3012, ← c3013] at e_78_12
  have hr := e_78_12
  simp only [e_78_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f580 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) := by
  have c2996 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_10 := arithEq_of_rows h (i := 10) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2996, ← c2997, ← c2998] at e_42_10
  have e_73_4 := arithEq_of_rows h (i := 4) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c2999, ← c3000, k3, ← c3001] at e_73_4
  have hr := e_73_4
  simp only [e_42_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f581 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 78 59) = bselect (a (.wire 73 19)) (a (.virt 80)) (a (.virt 84)) := by
  have c3014 := (leafCircuit_copies94 a h).2.2.2.2.2.2.1
  have c3015 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.1
  have c3016 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.1
  have c3017 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.1
  have c3018 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3019 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_78_13 := arithEq_of_rows h (i := 13) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_13
  simp only [← c3014, ← c3015, ← c3016] at e_78_13
  have e_78_14 := arithEq_of_rows h (i := 14) leafCircuit_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_14
  simp only [← c3017, ← c3018, ← c3019] at e_78_14
  have hr := e_78_14
  simp only [e_78_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f582 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 7) = bselect (a (.virt 340)) (a (.wire 68 31)) (a (.wire 78 59)) := by
  have c3020 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3021 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3022 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3023 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3024 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3025 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_0 := arithEq_of_rows h (i := 0) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_0
  simp only [← c3020, ← c3021, ← c3022] at e_79_0
  have e_79_1 := arithEq_of_rows h (i := 1) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_1
  simp only [← c3023, ← c3024, ← c3025] at e_79_1
  have hr := e_79_1
  simp only [e_79_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f583 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) := by
  have c2996 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_10 := arithEq_of_rows h (i := 10) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2996, ← c2997, ← c2998] at e_42_10
  have e_73_4 := arithEq_of_rows h (i := 4) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c2999, ← c3000, k3, ← c3001] at e_73_4
  have hr := e_73_4
  simp only [e_42_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f584 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 15) = bselect (a (.wire 73 19)) (a (.virt 81)) (a (.virt 85)) := by
  have c3026 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3027 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3028 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3029 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3030 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3031 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_2 := arithEq_of_rows h (i := 2) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_2
  simp only [← c3026, ← c3027, ← c3028] at e_79_2
  have e_79_3 := arithEq_of_rows h (i := 3) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_3
  simp only [← c3029, ← c3030, ← c3031] at e_79_3
  have hr := e_79_3
  simp only [e_79_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f585 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 23) = bselect (a (.virt 340)) (a (.wire 68 39)) (a (.wire 79 15)) := by
  have c3032 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3033 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3034 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3035 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3036 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3037 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_4 := arithEq_of_rows h (i := 4) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_4
  simp only [← c3032, ← c3033, ← c3034] at e_79_4
  have e_79_5 := arithEq_of_rows h (i := 5) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_5
  simp only [← c3035, ← c3036, ← c3037] at e_79_5
  have hr := e_79_5
  simp only [e_79_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f586 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) := by
  have c2996 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (leafCircuit_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_10 := arithEq_of_rows h (i := 10) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2996, ← c2997, ← c2998] at e_42_10
  have e_73_4 := arithEq_of_rows h (i := 4) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c2999, ← c3000, k3, ← c3001] at e_73_4
  have hr := e_73_4
  simp only [e_42_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f587 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 31) = bselect (a (.wire 73 19)) (a (.virt 82)) (a (.virt 86)) := by
  have c3038 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3039 := (leafCircuit_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3040 := (leafCircuit_copies95 a h).1
  have c3041 := (leafCircuit_copies95 a h).2.1
  have c3042 := (leafCircuit_copies95 a h).2.2.1
  have c3043 := (leafCircuit_copies95 a h).2.2.2.1
  have e_79_6 := arithEq_of_rows h (i := 6) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_6
  simp only [← c3038, ← c3039, ← c3040] at e_79_6
  have e_79_7 := arithEq_of_rows h (i := 7) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_7
  simp only [← c3041, ← c3042, ← c3043] at e_79_7
  have hr := e_79_7
  simp only [e_79_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f588 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 39) = bselect (a (.virt 340)) (a (.wire 68 47)) (a (.wire 79 31)) := by
  have c3044 := (leafCircuit_copies95 a h).2.2.2.2.1
  have c3045 := (leafCircuit_copies95 a h).2.2.2.2.2.1
  have c3046 := (leafCircuit_copies95 a h).2.2.2.2.2.2.1
  have c3047 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.1
  have c3048 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.1
  have c3049 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.1
  have e_79_8 := arithEq_of_rows h (i := 8) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_8
  simp only [← c3044, ← c3045, ← c3046] at e_79_8
  have e_79_9 := arithEq_of_rows h (i := 9) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_9
  simp only [← c3047, ← c3048, ← c3049] at e_79_9
  have hr := e_79_9
  simp only [e_79_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f589 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 47) = bselect (a (.virt 342)) (a (.wire 68 23)) (a (.virt 83)) := by
  have c3050 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3051 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3052 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3053 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3054 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3055 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_10 := arithEq_of_rows h (i := 10) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_10
  simp only [← c3050, ← c3051, ← c3052] at e_79_10
  have e_79_11 := arithEq_of_rows h (i := 11) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_11
  simp only [← c3053, ← c3054, ← c3055] at e_79_11
  have hr := e_79_11
  simp only [e_79_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f590 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 79 55) = bselect (a (.virt 342)) (a (.wire 68 31)) (a (.virt 84)) := by
  have c3056 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3057 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3058 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3059 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3060 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3061 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_12 := arithEq_of_rows h (i := 12) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_12
  simp only [← c3056, ← c3057, ← c3058] at e_79_12
  have e_79_13 := arithEq_of_rows h (i := 13) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_13
  simp only [← c3059, ← c3060, ← c3061] at e_79_13
  have hr := e_79_13
  simp only [e_79_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f591 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 3) = bselect (a (.virt 342)) (a (.wire 68 39)) (a (.virt 85)) := by
  have c3062 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3063 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3064 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3065 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3066 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3067 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_14 := arithEq_of_rows h (i := 14) leafCircuit_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_14
  simp only [← c3062, ← c3063, ← c3064] at e_79_14
  have e_80_0 := arithEq_of_rows h (i := 0) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_0
  simp only [← c3065, ← c3066, ← c3067] at e_80_0
  have hr := e_80_0
  simp only [e_79_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f592 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 11) = bselect (a (.virt 342)) (a (.wire 68 47)) (a (.virt 86)) := by
  have c3068 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3069 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3070 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3071 := (leafCircuit_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3072 := (leafCircuit_copies96 a h).1
  have c3073 := (leafCircuit_copies96 a h).2.1
  have e_80_1 := arithEq_of_rows h (i := 1) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_1
  simp only [← c3068, ← c3069, ← c3070] at e_80_1
  have e_80_2 := arithEq_of_rows h (i := 2) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_2
  simp only [← c3071, ← c3072, ← c3073] at e_80_2
  have hr := e_80_2
  simp only [e_80_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f593 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 83 12) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 0 ∧ a (.wire 83 13) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 1 ∧ a (.wire 83 14) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 2 ∧ a (.wire 83 15) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 3) := by
  have c3074 := (leafCircuit_copies96 a h).2.2.1
  have c3075 := (leafCircuit_copies96 a h).2.2.2.1
  have c3076 := (leafCircuit_copies96 a h).2.2.2.2.1
  have c3077 := (leafCircuit_copies96 a h).2.2.2.2.2.1
  have c3078 := (leafCircuit_copies96 a h).2.2.2.2.2.2.1
  have c3079 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.1
  have c3080 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.1
  have c3081 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.1
  have c3082 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3083 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3084 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3085 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3086 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3087 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3088 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3089 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3090 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3091 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3092 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3093 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3094 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3095 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3096 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3097 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3098 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3099 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3100 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3101 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3102 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3103 := (leafCircuit_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3104 := (leafCircuit_copies97 a h).1
  have c3105 := (leafCircuit_copies97 a h).2.1
  have c3106 := (leafCircuit_copies97 a h).2.2.1
  have c3107 := (leafCircuit_copies97 a h).2.2.2.1
  have c3108 := (leafCircuit_copies97 a h).2.2.2.2.1
  have c3109 := (leafCircuit_copies97 a h).2.2.2.2.2.1
  have c3110 := (leafCircuit_copies97 a h).2.2.2.2.2.2.1
  have c3111 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.1
  have c3112 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.1
  have c3113 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.1
  have c3114 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3115 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3116 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3117 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3118 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3119 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3120 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3121 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3122 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3123 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3124 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3125 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3126 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3127 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3128 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3129 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3130 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3131 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3132 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3133 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3134 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3135 := (leafCircuit_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3136 := (leafCircuit_copies98 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_73_5 := arithEq_of_rows h (i := 5) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_5
  simp only [← c3086, ← c3087, k3, ← c3088] at e_73_5
  have e_73_6 := arithEq_of_rows h (i := 6) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_6
  simp only [← c3089, ← c3090, k3, ← c3091] at e_73_6
  have e_73_7 := arithEq_of_rows h (i := 7) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_7
  simp only [← c3092, ← c3093, k3, ← c3094] at e_73_7
  have e_73_8 := arithEq_of_rows h (i := 8) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_8
  simp only [← c3095, ← c3096, k3, ← c3097] at e_73_8
  have e_73_9 := arithEq_of_rows h (i := 9) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_9
  simp only [← c3098, ← c3099, k3, ← c3100] at e_73_9
  have e_73_10 := arithEq_of_rows h (i := 10) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_10
  simp only [← c3101, ← c3102, k3, ← c3103] at e_73_10
  have e_73_11 := arithEq_of_rows h (i := 11) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_11
  simp only [← c3104, ← c3105, k3, ← c3106] at e_73_11
  have e_73_12 := arithEq_of_rows h (i := 12) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_12
  simp only [← c3107, ← c3108, k3, ← c3109] at e_73_12
  have e_73_13 := arithEq_of_rows h (i := 13) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_13
  simp only [← c3122, ← c3123, k3, ← c3124, k3] at e_73_13
  have hin0 : poseidon2In a 81 = addBlock (fun _ => 0) [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35)] :=
    poseidon2In_first c3074.symm c3075.symm c3076.symm c3077.symm c3078.symm c3079.symm c3080.symm c3081.symm (c3082.symm.trans k0) (c3083.symm.trans k0) (c3084.symm.trans k0) (c3085.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 81) rfl rfl hin0
  have hin1 : poseidon2In a 82 = addBlock (poseidon2Out a 81) [a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] :=
    poseidon2In_chain (by linear_combination -c3110 + e_73_5) (by linear_combination -c3111 + e_73_6) (by linear_combination -c3112 + e_73_7) (by linear_combination -c3113 + e_73_8) (by linear_combination -c3114 + e_73_9) (by linear_combination -c3115 + e_73_10) (by linear_combination -c3116 + e_73_11) (by linear_combination -c3117 + e_73_12) c3118.symm c3119.symm c3120.symm c3121.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 82) rfl rfl hin1
  have hin2 : poseidon2In a 83 = addBlock (poseidon2Out a 82) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c3125 + e_73_13) (c3126.symm.trans (add_zero _).symm) (c3127.symm.trans (add_zero _).symm) (c3128.symm.trans (add_zero _).symm) (c3129.symm.trans (add_zero _).symm) (c3130.symm.trans (add_zero _).symm) (c3131.symm.trans (add_zero _).symm) (c3132.symm.trans (add_zero _).symm) c3133.symm c3134.symm c3135.symm c3136.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 83) rfl rfl hin2
  have hpad : pad10 [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] =
      [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35)] ++ ([a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f594 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 19) = bselect (a (.wire 73 11)) (a (.wire 83 12)) (a (.wire 68 23)) := by
  have c3137 := (leafCircuit_copies98 a h).2.1
  have c3138 := (leafCircuit_copies98 a h).2.2.1
  have c3139 := (leafCircuit_copies98 a h).2.2.2.1
  have c3140 := (leafCircuit_copies98 a h).2.2.2.2.1
  have c3141 := (leafCircuit_copies98 a h).2.2.2.2.2.1
  have c3142 := (leafCircuit_copies98 a h).2.2.2.2.2.2.1
  have e_80_3 := arithEq_of_rows h (i := 3) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_3
  simp only [← c3137, ← c3138, ← c3139] at e_80_3
  have e_80_4 := arithEq_of_rows h (i := 4) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_4
  simp only [← c3140, ← c3141, ← c3142] at e_80_4
  have hr := e_80_4
  simp only [e_80_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f595 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 27) = bselect (a (.wire 73 11)) (a (.wire 83 13)) (a (.wire 68 31)) := by
  have c3143 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.1
  have c3144 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.1
  have c3145 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.1
  have c3146 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3147 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3148 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_5 := arithEq_of_rows h (i := 5) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_5
  simp only [← c3143, ← c3144, ← c3145] at e_80_5
  have e_80_6 := arithEq_of_rows h (i := 6) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_6
  simp only [← c3146, ← c3147, ← c3148] at e_80_6
  have hr := e_80_6
  simp only [e_80_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f596 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 35) = bselect (a (.wire 73 11)) (a (.wire 83 14)) (a (.wire 68 39)) := by
  have c3149 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3150 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3151 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3152 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3153 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3154 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_7 := arithEq_of_rows h (i := 7) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_7
  simp only [← c3149, ← c3150, ← c3151] at e_80_7
  have e_80_8 := arithEq_of_rows h (i := 8) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_8
  simp only [← c3152, ← c3153, ← c3154] at e_80_8
  have hr := e_80_8
  simp only [e_80_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f597 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 43) = bselect (a (.wire 73 11)) (a (.wire 83 15)) (a (.wire 68 47)) := by
  have c3155 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3156 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3157 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3158 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3159 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3160 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_80_9 := arithEq_of_rows h (i := 9) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_9
  simp only [← c3155, ← c3156, ← c3157] at e_80_9
  have e_80_10 := arithEq_of_rows h (i := 10) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_10
  simp only [← c3158, ← c3159, ← c3160] at e_80_10
  have hr := e_80_10
  simp only [e_80_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f598 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 84 1), a (.wire 84 2), a (.wire 84 3), a (.wire 84 4), a (.wire 84 5)] := by
  have c3161 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3162 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3163 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3164 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3165 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3166 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3167 := (leafCircuit_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3168 := (leafCircuit_copies99 a h).1
  have c3169 := (leafCircuit_copies99 a h).2.1
  have c3170 := (leafCircuit_copies99 a h).2.2.1
  have c3171 := (leafCircuit_copies99 a h).2.2.2.1
  have c3172 := (leafCircuit_copies99 a h).2.2.2.2.1
  have c3173 := (leafCircuit_copies99 a h).2.2.2.2.2.1
  have c3174 := (leafCircuit_copies99 a h).2.2.2.2.2.2.1
  have c3175 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.1
  have c3176 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.1
  have c3177 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.1
  have c3178 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3179 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3180 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3181 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3182 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3183 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3184 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3185 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3186 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3187 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3188 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3189 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3190 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3191 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3192 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3193 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3194 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3195 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3196 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3197 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3198 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3199 := (leafCircuit_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3200 := (leafCircuit_copies100 a h).1
  have c3201 := (leafCircuit_copies100 a h).2.1
  have c3202 := (leafCircuit_copies100 a h).2.2.1
  have c3203 := (leafCircuit_copies100 a h).2.2.2.1
  have c3204 := (leafCircuit_copies100 a h).2.2.2.2.1
  have c3205 := (leafCircuit_copies100 a h).2.2.2.2.2.1
  have c3206 := (leafCircuit_copies100 a h).2.2.2.2.2.2.1
  have c3207 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.1
  have c3208 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.1
  have c3209 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.1
  have c3210 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3211 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3212 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3213 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3214 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3215 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 84) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c3161.trans k0
    · exact c3162.trans k0
    · exact c3163.trans k0
    · exact c3164.trans k0
    · exact c3165.trans k0
    · exact c3166.trans k0
    · exact c3167.trans k0
    · exact c3168.trans k0
    · exact c3169.trans k0
    · exact c3170.trans k0
    · exact c3171.trans k0
    · exact c3172.trans k0
    · exact c3173.trans k0
    · exact c3174.trans k0
    · exact c3175.trans k0
    · exact c3176.trans k0
    · exact c3177.trans k0
    · exact c3178.trans k0
    · exact c3179.trans k0
    · exact c3180.trans k0
    · exact c3181.trans k0
    · exact c3182.trans k0
    · exact c3183.trans k0
    · exact c3184.trans k0
    · exact c3185.trans k0
    · exact c3186.trans k0
    · exact c3187.trans k0
    · exact c3188.trans k0
    · exact c3189.trans k0
    · exact c3190.trans k0
    · exact c3191.trans k0
    · exact c3192.trans k0
    · exact c3193.trans k0
    · exact c3194.trans k0
    · exact c3195.trans k0
    · exact c3196.trans k0
    · exact c3197.trans k0
    · exact c3198.trans k0
    · exact c3199.trans k0
    · exact c3200.trans k0
    · exact c3201.trans k0
    · exact c3202.trans k0
    · exact c3203.trans k0
    · exact c3204.trans k0
    · exact c3205.trans k0
    · exact c3206.trans k0
    · exact c3207.trans k0
    · exact c3208.trans k0
    · exact c3209.trans k0
    · exact c3210.trans k0
    · exact c3211.trans k0
    · exact c3212.trans k0
    · exact c3213.trans k0
    · exact c3214.trans k0
    )
  rwa [c3215] at hr

theorem leafCircuit_f599 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f600 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 5) = band (a (.virt 292)) (a (.wire 84 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f601 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 5) = band (a (.wire 84 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f602 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 5) = bor (a (.virt 14)) (a (.wire 84 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f603 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 84 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f604 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f605 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 5) = a (.virt 14) + a (.wire 84 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f606 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 5) = a (.wire 84 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f607 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 47) = bnot (a (.wire 84 5)) := by
  have c3216 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3217 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3218 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_80_11 := arithEq_of_rows h (i := 11) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_11
  simp only [← c3216, k3, ← c3217, k3, ← c3218] at e_80_11
  have hr := e_80_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f608 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 47) = band (a (.virt 292)) (a (.wire 80 47)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f609 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f610 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 4) = band (a (.virt 292)) (a (.wire 84 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f611 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 3) = band (a (.wire 84 4)) (a (.wire 80 47)) := by
  have c3219 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3220 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3221 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_0 := arithEq_of_rows h (i := 0) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_0
  simp only [← c3219, ← c3220, ← c3221] at e_85_0
  have hr := e_85_0
  simp only [band]
  linear_combination hr

theorem leafCircuit_f612 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 59) = bor (a (.wire 84 5)) (a (.wire 85 3)) := by
  have c3222 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3223 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3224 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3225 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3226 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3227 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_11 := arithEq_of_rows h (i := 11) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_11
  simp only [← c3222, ← c3223, ← c3224] at e_42_11
  have e_73_14 := arithEq_of_rows h (i := 14) leafCircuit_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_14
  simp only [← c3225, ← c3226, k3, ← c3227] at e_73_14
  have hr := e_73_14
  simp only [e_42_11] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f613 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 84 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f614 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f615 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 4) = a (.virt 14) + a (.wire 84 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f616 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 4) = a (.wire 84 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f617 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 51) = bnot (a (.wire 84 4)) := by
  have c3228 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3229 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3230 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_80_12 := arithEq_of_rows h (i := 12) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_12
  simp only [← c3228, k3, ← c3229, k3, ← c3230] at e_80_12
  have hr := e_80_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f618 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 7) = band (a (.wire 80 47)) (a (.wire 80 51)) := by
  have c3231 := (leafCircuit_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3232 := (leafCircuit_copies101 a h).1
  have c3233 := (leafCircuit_copies101 a h).2.1
  have e_85_1 := arithEq_of_rows h (i := 1) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_1
  simp only [← c3231, ← c3232, ← c3233] at e_85_1
  have hr := e_85_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f619 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f620 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 84 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f621 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 85 7)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f622 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 59) = bor (a (.wire 73 59)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f623 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 3) = a (.virt 292) * a (.wire 84 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f624 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 11) = a (.virt 294) * a (.wire 84 3) := by
  have c3234 := (leafCircuit_copies101 a h).2.2.1
  have c3235 := (leafCircuit_copies101 a h).2.2.2.1
  have c3236 := (leafCircuit_copies101 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_85_2 := arithEq_of_rows h (i := 2) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_2
  simp only [← c3234, k5, ← c3235, ← c3236, k5] at e_85_2
  have hr := e_85_2
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f625 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 3) = a (.virt 292) + a (.wire 84 3) := by
  have c3237 := (leafCircuit_copies101 a h).2.2.2.2.2.1
  have c3238 := (leafCircuit_copies101 a h).2.2.2.2.2.2.1
  have c3239 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_86_0 := arithEq_of_rows h (i := 0) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_0
  simp only [← c3237, k3, ← c3238, k3, ← c3239] at e_86_0
  have hr := e_86_0
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f626 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 55) = a (.wire 86 3) - a (.wire 85 11) := by
  have c3240 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.1
  have c3241 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.1
  have c3242 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_80_13 := arithEq_of_rows h (i := 13) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_13
  simp only [← c3240, ← c3241, k3, ← c3242] at e_80_13
  have hr := e_80_13
  linear_combination hr

theorem leafCircuit_f627 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 80 59) = bnot (a (.wire 80 55)) := by
  have c3243 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3244 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3245 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_80_14 := arithEq_of_rows h (i := 14) leafCircuit_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_14
  simp only [← c3243, k3, ← c3244, k3, ← c3245] at e_80_14
  have hr := e_80_14
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f628 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 15) = band (a (.wire 85 7)) (a (.wire 80 59)) := by
  have c3246 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3247 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3248 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_3 := arithEq_of_rows h (i := 3) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_3
  simp only [← c3246, ← c3247, ← c3248] at e_85_3
  have hr := e_85_3
  simp only [band]
  linear_combination hr

theorem leafCircuit_f629 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f630 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 84 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f631 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 85 15)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f632 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 73 59) = bor (a (.wire 73 59)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f633 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 2) = a (.virt 292) * a (.wire 84 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f634 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 19) = a (.virt 294) * a (.wire 84 2) := by
  have c3249 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3250 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3251 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_85_4 := arithEq_of_rows h (i := 4) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_4
  simp only [← c3249, k5, ← c3250, ← c3251, k5] at e_85_4
  have hr := e_85_4
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f635 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 7) = a (.virt 292) + a (.wire 84 2) := by
  have c3252 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3253 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3254 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_86_1 := arithEq_of_rows h (i := 1) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_1
  simp only [← c3252, k3, ← c3253, k3, ← c3254] at e_86_1
  have hr := e_86_1
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f636 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 87 3) = a (.wire 86 7) - a (.wire 85 19) := by
  have c3255 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3256 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3257 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_87_0 := arithEq_of_rows h (i := 0) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_0
  simp only [← c3255, ← c3256, k3, ← c3257] at e_87_0
  have hr := e_87_0
  linear_combination hr

theorem leafCircuit_f637 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 87 7) = bnot (a (.wire 87 3)) := by
  have c3258 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3259 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3260 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_87_1 := arithEq_of_rows h (i := 1) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_1
  simp only [← c3258, k3, ← c3259, k3, ← c3260] at e_87_1
  have hr := e_87_1
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f638 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 23) = band (a (.wire 85 15)) (a (.wire 87 7)) := by
  have c3261 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3262 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3263 := (leafCircuit_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_85_5 := arithEq_of_rows h (i := 5) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_5
  simp only [← c3261, ← c3262, ← c3263] at e_85_5
  have hr := e_85_5
  simp only [band]
  linear_combination hr

theorem leafCircuit_f639 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f640 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 1) = band (a (.virt 292)) (a (.wire 84 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f641 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 27) = band (a (.wire 84 1)) (a (.wire 85 23)) := by
  have c3264 := (leafCircuit_copies102 a h).1
  have c3265 := (leafCircuit_copies102 a h).2.1
  have c3266 := (leafCircuit_copies102 a h).2.2.1
  have e_85_6 := arithEq_of_rows h (i := 6) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_6
  simp only [← c3264, ← c3265, ← c3266] at e_85_6
  have hr := e_85_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f642 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 11) = bor (a (.wire 73 59)) (a (.wire 85 27)) := by
  have c3267 := (leafCircuit_copies102 a h).2.2.2.1
  have c3268 := (leafCircuit_copies102 a h).2.2.2.2.1
  have c3269 := (leafCircuit_copies102 a h).2.2.2.2.2.1
  have c3270 := (leafCircuit_copies102 a h).2.2.2.2.2.2.1
  have c3271 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.1
  have c3272 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_12 := arithEq_of_rows h (i := 12) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_12
  simp only [← c3267, ← c3268, ← c3269] at e_42_12
  have e_86_2 := arithEq_of_rows h (i := 2) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_2
  simp only [← c3270, ← c3271, k3, ← c3272] at e_86_2
  have hr := e_86_2
  simp only [e_42_12] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f643 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 84 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f644 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f645 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 1) = a (.virt 14) + a (.wire 84 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f646 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 84 1) = a (.wire 84 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f647 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 87 11) = bnot (a (.wire 84 1)) := by
  have c3273 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.1
  have c3274 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3275 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_87_2 := arithEq_of_rows h (i := 2) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_2
  simp only [← c3273, k3, ← c3274, k3, ← c3275] at e_87_2
  have hr := e_87_2
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f648 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 85 31) = band (a (.wire 85 23)) (a (.wire 87 11)) := by
  have c3276 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3277 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3278 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_7 := arithEq_of_rows h (i := 7) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_7
  simp only [← c3276, ← c3277, ← c3278] at e_85_7
  have hr := e_85_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f649 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 213)) 2 := by
  have c3279 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3280 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3281 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3282 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3283 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3284 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3285 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3286 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3287 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3288 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3289 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3290 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3291 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3292 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3293 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3294 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3295 := (leafCircuit_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3296 := (leafCircuit_copies103 a h).1
  have c3297 := (leafCircuit_copies103 a h).2.1
  have c3298 := (leafCircuit_copies103 a h).2.2.1
  have c3299 := (leafCircuit_copies103 a h).2.2.2.1
  have c3300 := (leafCircuit_copies103 a h).2.2.2.2.1
  have c3301 := (leafCircuit_copies103 a h).2.2.2.2.2.1
  have c3302 := (leafCircuit_copies103 a h).2.2.2.2.2.2.1
  have c3303 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.1
  have c3304 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.1
  have c3305 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.1
  have c3306 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3307 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3308 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3309 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3310 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3311 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3312 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3313 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3314 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3315 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3316 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3317 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3318 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3319 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3320 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3321 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3322 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3323 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3324 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3325 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3326 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3327 := (leafCircuit_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3328 := (leafCircuit_copies104 a h).1
  have c3329 := (leafCircuit_copies104 a h).2.1
  have c3330 := (leafCircuit_copies104 a h).2.2.1
  have c3331 := (leafCircuit_copies104 a h).2.2.2.1
  have c3332 := (leafCircuit_copies104 a h).2.2.2.2.1
  have c3333 := (leafCircuit_copies104 a h).2.2.2.2.2.1
  have c3334 := (leafCircuit_copies104 a h).2.2.2.2.2.2.1
  have c3335 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.1
  have c3336 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 88) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c3279.trans k0
    · exact c3280.trans k0
    · exact c3281.trans k0
    · exact c3282.trans k0
    · exact c3283.trans k0
    · exact c3284.trans k0
    · exact c3285.trans k0
    · exact c3286.trans k0
    · exact c3287.trans k0
    · exact c3288.trans k0
    · exact c3289.trans k0
    · exact c3290.trans k0
    · exact c3291.trans k0
    · exact c3292.trans k0
    · exact c3293.trans k0
    · exact c3294.trans k0
    · exact c3295.trans k0
    · exact c3296.trans k0
    · exact c3297.trans k0
    · exact c3298.trans k0
    · exact c3299.trans k0
    · exact c3300.trans k0
    · exact c3301.trans k0
    · exact c3302.trans k0
    · exact c3303.trans k0
    · exact c3304.trans k0
    · exact c3305.trans k0
    · exact c3306.trans k0
    · exact c3307.trans k0
    · exact c3308.trans k0
    · exact c3309.trans k0
    · exact c3310.trans k0
    · exact c3311.trans k0
    · exact c3312.trans k0
    · exact c3313.trans k0
    · exact c3314.trans k0
    · exact c3315.trans k0
    · exact c3316.trans k0
    · exact c3317.trans k0
    · exact c3318.trans k0
    · exact c3319.trans k0
    · exact c3320.trans k0
    · exact c3321.trans k0
    · exact c3322.trans k0
    · exact c3323.trans k0
    · exact c3324.trans k0
    · exact c3325.trans k0
    · exact c3326.trans k0
    · exact c3327.trans k0
    · exact c3328.trans k0
    · exact c3329.trans k0
    · exact c3330.trans k0
    · exact c3331.trans k0
    · exact c3332.trans k0
    · exact c3333.trans k0
    · exact c3334.trans k0
    · exact c3335.trans k0
    )
  rwa [c3336] at hr

theorem leafCircuit_f650 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 213)) (a (.virt 14)) (a (.virt 344)) (a (.virt 345)) := by
  have c3337 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.1
  have c3338 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3339 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3340 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3341 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3342 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3343 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3344 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3345 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3346 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3347 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3348 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3349 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3350 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_85_8 := arithEq_of_rows h (i := 8) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_8
  simp only [← c3340, ← c3341, ← c3342] at e_85_8
  have e_85_9 := arithEq_of_rows h (i := 9) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_9
  simp only [← c3343, ← c3344, ← c3345] at e_85_9
  have e_87_3 := arithEq_of_rows h (i := 3) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_3
  simp only [← c3337, k3, ← c3338, k3, ← c3339] at e_87_3
  have e_87_4 := arithEq_of_rows h (i := 4) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_4
  simp only [← c3346, ← c3347, k3, ← c3348] at e_87_4
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_85_8
    linear_combination c3349.trans k0 - hc
  · have hc := e_87_4
    simp only [e_87_3, e_85_9] at hc
    linear_combination c3350.trans k0 - hc

theorem leafCircuit_f651 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 213)) (a (.virt 292)) (a (.virt 346)) (a (.virt 347)) := by
  have c3351 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3352 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3353 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3354 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3355 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3356 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3357 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3358 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3359 := (leafCircuit_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3360 := (leafCircuit_copies105 a h).1
  have c3361 := (leafCircuit_copies105 a h).2.1
  have c3362 := (leafCircuit_copies105 a h).2.2.1
  have c3363 := (leafCircuit_copies105 a h).2.2.2.1
  have c3364 := (leafCircuit_copies105 a h).2.2.2.2.1
  have c3365 := (leafCircuit_copies105 a h).2.2.2.2.2.1
  have c3366 := (leafCircuit_copies105 a h).2.2.2.2.2.2.1
  have c3367 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_85_10 := arithEq_of_rows h (i := 10) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_10
  simp only [← c3357, ← c3358, ← c3359] at e_85_10
  have e_85_11 := arithEq_of_rows h (i := 11) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_11
  simp only [← c3360, ← c3361, ← c3362] at e_85_11
  have e_87_5 := arithEq_of_rows h (i := 5) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_5
  simp only [← c3351, k3, ← c3352, k3, ← c3353] at e_87_5
  have e_87_6 := arithEq_of_rows h (i := 6) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_6
  simp only [← c3354, ← c3355, k3, ← c3356, k3] at e_87_6
  have e_87_7 := arithEq_of_rows h (i := 7) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_7
  simp only [← c3363, ← c3364, k3, ← c3365] at e_87_7
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_85_10
    simp only [e_87_6] at hc
    linear_combination c3366.trans k0 - hc
  · have hc := e_87_7
    simp only [e_87_5, e_85_11, e_87_6] at hc
    linear_combination c3367.trans k0 - hc

theorem leafCircuit_f652 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 213)) (a (.virt 294)) (a (.virt 348)) (a (.virt 349)) := by
  have c3368 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.1
  have c3369 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.1
  have c3370 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3371 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3372 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3373 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3374 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3375 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3376 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3377 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3378 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3379 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3380 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3381 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3382 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3383 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3384 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_85_12 := arithEq_of_rows h (i := 12) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_12
  simp only [← c3374, ← c3375, ← c3376] at e_85_12
  have e_85_13 := arithEq_of_rows h (i := 13) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_13
  simp only [← c3377, ← c3378, ← c3379] at e_85_13
  have e_87_8 := arithEq_of_rows h (i := 8) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_8
  simp only [← c3368, k3, ← c3369, k3, ← c3370] at e_87_8
  have e_87_9 := arithEq_of_rows h (i := 9) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_9
  simp only [← c3371, ← c3372, k3, ← c3373, k5] at e_87_9
  have e_87_10 := arithEq_of_rows h (i := 10) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_10
  simp only [← c3380, ← c3381, k3, ← c3382] at e_87_10
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_85_12
    simp only [e_87_9] at hc
    linear_combination c3383.trans k0 - hc
  · have hc := e_87_10
    simp only [e_87_8, e_85_13, e_87_9] at hc
    linear_combination c3384.trans k0 - hc

theorem leafCircuit_f653 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 213)) (a (.virt 295)) (a (.virt 350)) (a (.virt 351)) := by
  have c3385 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3386 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3387 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3388 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3389 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3390 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3391 := (leafCircuit_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3392 := (leafCircuit_copies106 a h).1
  have c3393 := (leafCircuit_copies106 a h).2.1
  have c3394 := (leafCircuit_copies106 a h).2.2.1
  have c3395 := (leafCircuit_copies106 a h).2.2.2.1
  have c3396 := (leafCircuit_copies106 a h).2.2.2.2.1
  have c3397 := (leafCircuit_copies106 a h).2.2.2.2.2.1
  have c3398 := (leafCircuit_copies106 a h).2.2.2.2.2.2.1
  have c3399 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.1
  have c3400 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.1
  have c3401 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_85_14 := arithEq_of_rows h (i := 14) leafCircuit_row85 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_14
  simp only [← c3391, ← c3392, ← c3393] at e_85_14
  have e_87_11 := arithEq_of_rows h (i := 11) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_11
  simp only [← c3385, k3, ← c3386, k3, ← c3387] at e_87_11
  have e_87_12 := arithEq_of_rows h (i := 12) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_12
  simp only [← c3388, ← c3389, k3, ← c3390, k6] at e_87_12
  have e_87_13 := arithEq_of_rows h (i := 13) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_13
  simp only [← c3397, ← c3398, k3, ← c3399] at e_87_13
  have e_89_0 := arithEq_of_rows h (i := 0) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_0
  simp only [← c3394, ← c3395, ← c3396] at e_89_0
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_85_14
    simp only [e_87_12] at hc
    linear_combination c3400.trans k0 - hc
  · have hc := e_87_13
    simp only [e_87_11, e_89_0, e_87_12] at hc
    linear_combination c3401.trans k0 - hc

theorem leafCircuit_f654 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 3) = bselect (a (.virt 344)) (a (.wire 80 19)) (a (.virt 87)) := by
  have c3402 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3403 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3404 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3405 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3406 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3407 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_14 := arithEq_of_rows h (i := 14) leafCircuit_row87 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_14
  simp only [← c3402, ← c3403, ← c3404] at e_87_14
  have e_90_0 := arithEq_of_rows h (i := 0) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_0
  simp only [← c3405, ← c3406, ← c3407] at e_90_0
  have hr := e_90_0
  simp only [e_87_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f655 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 11) = bselect (a (.virt 344)) (a (.wire 80 27)) (a (.virt 88)) := by
  have c3408 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3409 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3410 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3411 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3412 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3413 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_1 := arithEq_of_rows h (i := 1) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_1
  simp only [← c3408, ← c3409, ← c3410] at e_90_1
  have e_90_2 := arithEq_of_rows h (i := 2) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_2
  simp only [← c3411, ← c3412, ← c3413] at e_90_2
  have hr := e_90_2
  simp only [e_90_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f656 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 19) = bselect (a (.virt 344)) (a (.wire 80 35)) (a (.virt 89)) := by
  have c3414 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3415 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3416 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3417 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3418 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3419 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_3 := arithEq_of_rows h (i := 3) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_3
  simp only [← c3414, ← c3415, ← c3416] at e_90_3
  have e_90_4 := arithEq_of_rows h (i := 4) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_4
  simp only [← c3417, ← c3418, ← c3419] at e_90_4
  have hr := e_90_4
  simp only [e_90_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f657 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 27) = bselect (a (.virt 344)) (a (.wire 80 43)) (a (.virt 90)) := by
  have c3420 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3421 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3422 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3423 := (leafCircuit_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3424 := (leafCircuit_copies107 a h).1
  have c3425 := (leafCircuit_copies107 a h).2.1
  have e_90_5 := arithEq_of_rows h (i := 5) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_5
  simp only [← c3420, ← c3421, ← c3422] at e_90_5
  have e_90_6 := arithEq_of_rows h (i := 6) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_6
  simp only [← c3423, ← c3424, ← c3425] at e_90_6
  have hr := e_90_6
  simp only [e_90_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f658 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 35) = bselect (a (.virt 344)) (a (.virt 87)) (a (.virt 91)) := by
  have c3426 := (leafCircuit_copies107 a h).2.2.1
  have c3427 := (leafCircuit_copies107 a h).2.2.2.1
  have c3428 := (leafCircuit_copies107 a h).2.2.2.2.1
  have c3429 := (leafCircuit_copies107 a h).2.2.2.2.2.1
  have c3430 := (leafCircuit_copies107 a h).2.2.2.2.2.2.1
  have c3431 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.1
  have e_90_7 := arithEq_of_rows h (i := 7) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_7
  simp only [← c3426, ← c3427, ← c3428] at e_90_7
  have e_90_8 := arithEq_of_rows h (i := 8) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_8
  simp only [← c3429, ← c3430, ← c3431] at e_90_8
  have hr := e_90_8
  simp only [e_90_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f659 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 43) = bselect (a (.virt 346)) (a (.wire 80 19)) (a (.wire 90 35)) := by
  have c3432 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.1
  have c3433 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.1
  have c3434 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3435 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3436 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3437 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_9 := arithEq_of_rows h (i := 9) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_9
  simp only [← c3432, ← c3433, ← c3434] at e_90_9
  have e_90_10 := arithEq_of_rows h (i := 10) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_10
  simp only [← c3435, ← c3436, ← c3437] at e_90_10
  have hr := e_90_10
  simp only [e_90_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f660 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 51) = bselect (a (.virt 344)) (a (.virt 88)) (a (.virt 92)) := by
  have c3438 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3439 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3440 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3441 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3442 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3443 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_11 := arithEq_of_rows h (i := 11) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_11
  simp only [← c3438, ← c3439, ← c3440] at e_90_11
  have e_90_12 := arithEq_of_rows h (i := 12) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_12
  simp only [← c3441, ← c3442, ← c3443] at e_90_12
  have hr := e_90_12
  simp only [e_90_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f661 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 90 59) = bselect (a (.virt 346)) (a (.wire 80 27)) (a (.wire 90 51)) := by
  have c3444 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3445 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3446 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3447 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3448 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3449 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_13 := arithEq_of_rows h (i := 13) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_13
  simp only [← c3444, ← c3445, ← c3446] at e_90_13
  have e_90_14 := arithEq_of_rows h (i := 14) leafCircuit_row90 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_14
  simp only [← c3447, ← c3448, ← c3449] at e_90_14
  have hr := e_90_14
  simp only [e_90_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f662 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 7) = bselect (a (.virt 344)) (a (.virt 89)) (a (.virt 93)) := by
  have c3450 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3451 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3452 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3453 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3454 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3455 := (leafCircuit_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_91_0 := arithEq_of_rows h (i := 0) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_0
  simp only [← c3450, ← c3451, ← c3452] at e_91_0
  have e_91_1 := arithEq_of_rows h (i := 1) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_1
  simp only [← c3453, ← c3454, ← c3455] at e_91_1
  have hr := e_91_1
  simp only [e_91_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f663 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 15) = bselect (a (.virt 346)) (a (.wire 80 35)) (a (.wire 91 7)) := by
  have c3456 := (leafCircuit_copies108 a h).1
  have c3457 := (leafCircuit_copies108 a h).2.1
  have c3458 := (leafCircuit_copies108 a h).2.2.1
  have c3459 := (leafCircuit_copies108 a h).2.2.2.1
  have c3460 := (leafCircuit_copies108 a h).2.2.2.2.1
  have c3461 := (leafCircuit_copies108 a h).2.2.2.2.2.1
  have e_91_2 := arithEq_of_rows h (i := 2) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_2
  simp only [← c3456, ← c3457, ← c3458] at e_91_2
  have e_91_3 := arithEq_of_rows h (i := 3) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_3
  simp only [← c3459, ← c3460, ← c3461] at e_91_3
  have hr := e_91_3
  simp only [e_91_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f664 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 23) = bselect (a (.virt 344)) (a (.virt 90)) (a (.virt 94)) := by
  have c3462 := (leafCircuit_copies108 a h).2.2.2.2.2.2.1
  have c3463 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.1
  have c3464 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.1
  have c3465 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.1
  have c3466 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3467 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_4 := arithEq_of_rows h (i := 4) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_4
  simp only [← c3462, ← c3463, ← c3464] at e_91_4
  have e_91_5 := arithEq_of_rows h (i := 5) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_5
  simp only [← c3465, ← c3466, ← c3467] at e_91_5
  have hr := e_91_5
  simp only [e_91_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f665 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 31) = bselect (a (.virt 346)) (a (.wire 80 43)) (a (.wire 91 23)) := by
  have c3468 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3469 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3470 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3471 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3472 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3473 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_6 := arithEq_of_rows h (i := 6) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_6
  simp only [← c3468, ← c3469, ← c3470] at e_91_6
  have e_91_7 := arithEq_of_rows h (i := 7) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_7
  simp only [← c3471, ← c3472, ← c3473] at e_91_7
  have hr := e_91_7
  simp only [e_91_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f666 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) := by
  have c3474 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3479 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_13 := arithEq_of_rows h (i := 13) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c3474, ← c3475, ← c3476] at e_42_13
  have e_86_3 := arithEq_of_rows h (i := 3) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c3477, ← c3478, k3, ← c3479] at e_86_3
  have hr := e_86_3
  simp only [e_42_13] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f667 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 39) = bselect (a (.wire 86 15)) (a (.virt 91)) (a (.virt 95)) := by
  have c3480 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3481 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3482 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3483 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3484 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3485 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_8 := arithEq_of_rows h (i := 8) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_8
  simp only [← c3480, ← c3481, ← c3482] at e_91_8
  have e_91_9 := arithEq_of_rows h (i := 9) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_9
  simp only [← c3483, ← c3484, ← c3485] at e_91_9
  have hr := e_91_9
  simp only [e_91_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f668 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 47) = bselect (a (.virt 348)) (a (.wire 80 19)) (a (.wire 91 39)) := by
  have c3486 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3487 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3488 := (leafCircuit_copies109 a h).1
  have c3489 := (leafCircuit_copies109 a h).2.1
  have c3490 := (leafCircuit_copies109 a h).2.2.1
  have c3491 := (leafCircuit_copies109 a h).2.2.2.1
  have e_91_10 := arithEq_of_rows h (i := 10) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_10
  simp only [← c3486, ← c3487, ← c3488] at e_91_10
  have e_91_11 := arithEq_of_rows h (i := 11) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_11
  simp only [← c3489, ← c3490, ← c3491] at e_91_11
  have hr := e_91_11
  simp only [e_91_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f669 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) := by
  have c3474 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3479 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_13 := arithEq_of_rows h (i := 13) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c3474, ← c3475, ← c3476] at e_42_13
  have e_86_3 := arithEq_of_rows h (i := 3) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c3477, ← c3478, k3, ← c3479] at e_86_3
  have hr := e_86_3
  simp only [e_42_13] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f670 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 91 55) = bselect (a (.wire 86 15)) (a (.virt 92)) (a (.virt 96)) := by
  have c3492 := (leafCircuit_copies109 a h).2.2.2.2.1
  have c3493 := (leafCircuit_copies109 a h).2.2.2.2.2.1
  have c3494 := (leafCircuit_copies109 a h).2.2.2.2.2.2.1
  have c3495 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.1
  have c3496 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.1
  have c3497 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.1
  have e_91_12 := arithEq_of_rows h (i := 12) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_12
  simp only [← c3492, ← c3493, ← c3494] at e_91_12
  have e_91_13 := arithEq_of_rows h (i := 13) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_13
  simp only [← c3495, ← c3496, ← c3497] at e_91_13
  have hr := e_91_13
  simp only [e_91_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f671 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 3) = bselect (a (.virt 348)) (a (.wire 80 27)) (a (.wire 91 55)) := by
  have c3498 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3499 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3500 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3501 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3502 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3503 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_14 := arithEq_of_rows h (i := 14) leafCircuit_row91 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_14
  simp only [← c3498, ← c3499, ← c3500] at e_91_14
  have e_92_0 := arithEq_of_rows h (i := 0) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_0
  simp only [← c3501, ← c3502, ← c3503] at e_92_0
  have hr := e_92_0
  simp only [e_91_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f672 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) := by
  have c3474 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3479 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_13 := arithEq_of_rows h (i := 13) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c3474, ← c3475, ← c3476] at e_42_13
  have e_86_3 := arithEq_of_rows h (i := 3) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c3477, ← c3478, k3, ← c3479] at e_86_3
  have hr := e_86_3
  simp only [e_42_13] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f673 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 11) = bselect (a (.wire 86 15)) (a (.virt 93)) (a (.virt 97)) := by
  have c3504 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3505 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3506 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3507 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3508 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3509 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_1 := arithEq_of_rows h (i := 1) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_1
  simp only [← c3504, ← c3505, ← c3506] at e_92_1
  have e_92_2 := arithEq_of_rows h (i := 2) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_2
  simp only [← c3507, ← c3508, ← c3509] at e_92_2
  have hr := e_92_2
  simp only [e_92_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f674 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 19) = bselect (a (.virt 348)) (a (.wire 80 35)) (a (.wire 92 11)) := by
  have c3510 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3511 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3512 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3513 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3514 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3515 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_3 := arithEq_of_rows h (i := 3) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_3
  simp only [← c3510, ← c3511, ← c3512] at e_92_3
  have e_92_4 := arithEq_of_rows h (i := 4) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_4
  simp only [← c3513, ← c3514, ← c3515] at e_92_4
  have hr := e_92_4
  simp only [e_92_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f675 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) := by
  have c3474 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3479 := (leafCircuit_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_13 := arithEq_of_rows h (i := 13) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c3474, ← c3475, ← c3476] at e_42_13
  have e_86_3 := arithEq_of_rows h (i := 3) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c3477, ← c3478, k3, ← c3479] at e_86_3
  have hr := e_86_3
  simp only [e_42_13] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f676 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 27) = bselect (a (.wire 86 15)) (a (.virt 94)) (a (.virt 98)) := by
  have c3516 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3517 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3518 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3519 := (leafCircuit_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3520 := (leafCircuit_copies110 a h).1
  have c3521 := (leafCircuit_copies110 a h).2.1
  have e_92_5 := arithEq_of_rows h (i := 5) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_5
  simp only [← c3516, ← c3517, ← c3518] at e_92_5
  have e_92_6 := arithEq_of_rows h (i := 6) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_6
  simp only [← c3519, ← c3520, ← c3521] at e_92_6
  have hr := e_92_6
  simp only [e_92_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f677 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 35) = bselect (a (.virt 348)) (a (.wire 80 43)) (a (.wire 92 27)) := by
  have c3522 := (leafCircuit_copies110 a h).2.2.1
  have c3523 := (leafCircuit_copies110 a h).2.2.2.1
  have c3524 := (leafCircuit_copies110 a h).2.2.2.2.1
  have c3525 := (leafCircuit_copies110 a h).2.2.2.2.2.1
  have c3526 := (leafCircuit_copies110 a h).2.2.2.2.2.2.1
  have c3527 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.1
  have e_92_7 := arithEq_of_rows h (i := 7) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_7
  simp only [← c3522, ← c3523, ← c3524] at e_92_7
  have e_92_8 := arithEq_of_rows h (i := 8) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_8
  simp only [← c3525, ← c3526, ← c3527] at e_92_8
  have hr := e_92_8
  simp only [e_92_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f678 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 43) = bselect (a (.virt 350)) (a (.wire 80 19)) (a (.virt 95)) := by
  have c3528 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.1
  have c3529 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.1
  have c3530 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3531 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3532 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3533 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_9 := arithEq_of_rows h (i := 9) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_9
  simp only [← c3528, ← c3529, ← c3530] at e_92_9
  have e_92_10 := arithEq_of_rows h (i := 10) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_10
  simp only [← c3531, ← c3532, ← c3533] at e_92_10
  have hr := e_92_10
  simp only [e_92_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f679 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 51) = bselect (a (.virt 350)) (a (.wire 80 27)) (a (.virt 96)) := by
  have c3534 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3535 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3536 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3537 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3538 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3539 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_11 := arithEq_of_rows h (i := 11) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_11
  simp only [← c3534, ← c3535, ← c3536] at e_92_11
  have e_92_12 := arithEq_of_rows h (i := 12) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_12
  simp only [← c3537, ← c3538, ← c3539] at e_92_12
  have hr := e_92_12
  simp only [e_92_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f680 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 92 59) = bselect (a (.virt 350)) (a (.wire 80 35)) (a (.virt 97)) := by
  have c3540 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3541 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3542 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3543 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3544 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3545 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_13 := arithEq_of_rows h (i := 13) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_13
  simp only [← c3540, ← c3541, ← c3542] at e_92_13
  have e_92_14 := arithEq_of_rows h (i := 14) leafCircuit_row92 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_14
  simp only [← c3543, ← c3544, ← c3545] at e_92_14
  have hr := e_92_14
  simp only [e_92_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f681 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 7) = bselect (a (.virt 350)) (a (.wire 80 43)) (a (.virt 98)) := by
  have c3546 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3547 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3548 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3549 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3550 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3551 := (leafCircuit_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_93_0 := arithEq_of_rows h (i := 0) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_0
  simp only [← c3546, ← c3547, ← c3548] at e_93_0
  have e_93_1 := arithEq_of_rows h (i := 1) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_1
  simp only [← c3549, ← c3550, ← c3551] at e_93_1
  have hr := e_93_1
  simp only [e_93_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f682 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 96 12) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 0 ∧ a (.wire 96 13) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 1 ∧ a (.wire 96 14) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 2 ∧ a (.wire 96 15) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 3) := by
  have c3552 := (leafCircuit_copies111 a h).1
  have c3553 := (leafCircuit_copies111 a h).2.1
  have c3554 := (leafCircuit_copies111 a h).2.2.1
  have c3555 := (leafCircuit_copies111 a h).2.2.2.1
  have c3556 := (leafCircuit_copies111 a h).2.2.2.2.1
  have c3557 := (leafCircuit_copies111 a h).2.2.2.2.2.1
  have c3558 := (leafCircuit_copies111 a h).2.2.2.2.2.2.1
  have c3559 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.1
  have c3560 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.1
  have c3561 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.1
  have c3562 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3563 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3564 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3565 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3566 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3567 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3568 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3569 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3570 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3571 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3572 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3573 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3574 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3575 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3576 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3577 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3578 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3579 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3580 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3581 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3582 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3583 := (leafCircuit_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3584 := (leafCircuit_copies112 a h).1
  have c3585 := (leafCircuit_copies112 a h).2.1
  have c3586 := (leafCircuit_copies112 a h).2.2.1
  have c3587 := (leafCircuit_copies112 a h).2.2.2.1
  have c3588 := (leafCircuit_copies112 a h).2.2.2.2.1
  have c3589 := (leafCircuit_copies112 a h).2.2.2.2.2.1
  have c3590 := (leafCircuit_copies112 a h).2.2.2.2.2.2.1
  have c3591 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.1
  have c3592 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.1
  have c3593 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.1
  have c3594 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3595 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3596 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3597 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3598 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3599 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3600 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3601 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3602 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3603 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3604 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3605 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3606 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3607 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3608 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3609 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3610 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3611 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3612 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3613 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3614 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_86_4 := arithEq_of_rows h (i := 4) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_4
  simp only [← c3564, ← c3565, k3, ← c3566] at e_86_4
  have e_86_5 := arithEq_of_rows h (i := 5) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_5
  simp only [← c3567, ← c3568, k3, ← c3569] at e_86_5
  have e_86_6 := arithEq_of_rows h (i := 6) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_6
  simp only [← c3570, ← c3571, k3, ← c3572] at e_86_6
  have e_86_7 := arithEq_of_rows h (i := 7) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_7
  simp only [← c3573, ← c3574, k3, ← c3575] at e_86_7
  have e_86_8 := arithEq_of_rows h (i := 8) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_8
  simp only [← c3576, ← c3577, k3, ← c3578] at e_86_8
  have e_86_9 := arithEq_of_rows h (i := 9) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_9
  simp only [← c3579, ← c3580, k3, ← c3581] at e_86_9
  have e_86_10 := arithEq_of_rows h (i := 10) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_10
  simp only [← c3582, ← c3583, k3, ← c3584] at e_86_10
  have e_86_11 := arithEq_of_rows h (i := 11) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_11
  simp only [← c3585, ← c3586, k3, ← c3587] at e_86_11
  have e_86_12 := arithEq_of_rows h (i := 12) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_12
  simp only [← c3600, ← c3601, k3, ← c3602, k3] at e_86_12
  have hin0 : poseidon2In a 94 = addBlock (fun _ => 0) [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31)] :=
    poseidon2In_first c3552.symm c3553.symm c3554.symm c3555.symm c3556.symm c3557.symm c3558.symm c3559.symm (c3560.symm.trans k0) (c3561.symm.trans k0) (c3562.symm.trans k0) (c3563.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 94) rfl rfl hin0
  have hin1 : poseidon2In a 95 = addBlock (poseidon2Out a 94) [a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] :=
    poseidon2In_chain (by linear_combination -c3588 + e_86_4) (by linear_combination -c3589 + e_86_5) (by linear_combination -c3590 + e_86_6) (by linear_combination -c3591 + e_86_7) (by linear_combination -c3592 + e_86_8) (by linear_combination -c3593 + e_86_9) (by linear_combination -c3594 + e_86_10) (by linear_combination -c3595 + e_86_11) c3596.symm c3597.symm c3598.symm c3599.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 95) rfl rfl hin1
  have hin2 : poseidon2In a 96 = addBlock (poseidon2Out a 95) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c3603 + e_86_12) (c3604.symm.trans (add_zero _).symm) (c3605.symm.trans (add_zero _).symm) (c3606.symm.trans (add_zero _).symm) (c3607.symm.trans (add_zero _).symm) (c3608.symm.trans (add_zero _).symm) (c3609.symm.trans (add_zero _).symm) (c3610.symm.trans (add_zero _).symm) c3611.symm c3612.symm c3613.symm c3614.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 96) rfl rfl hin2
  have hpad : pad10 [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] =
      [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31)] ++ ([a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f683 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 15) = bselect (a (.wire 86 11)) (a (.wire 96 12)) (a (.wire 80 19)) := by
  have c3615 := (leafCircuit_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3616 := (leafCircuit_copies113 a h).1
  have c3617 := (leafCircuit_copies113 a h).2.1
  have c3618 := (leafCircuit_copies113 a h).2.2.1
  have c3619 := (leafCircuit_copies113 a h).2.2.2.1
  have c3620 := (leafCircuit_copies113 a h).2.2.2.2.1
  have e_93_2 := arithEq_of_rows h (i := 2) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_2
  simp only [← c3615, ← c3616, ← c3617] at e_93_2
  have e_93_3 := arithEq_of_rows h (i := 3) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_3
  simp only [← c3618, ← c3619, ← c3620] at e_93_3
  have hr := e_93_3
  simp only [e_93_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f684 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 23) = bselect (a (.wire 86 11)) (a (.wire 96 13)) (a (.wire 80 27)) := by
  have c3621 := (leafCircuit_copies113 a h).2.2.2.2.2.1
  have c3622 := (leafCircuit_copies113 a h).2.2.2.2.2.2.1
  have c3623 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.1
  have c3624 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.1
  have c3625 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.1
  have c3626 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_93_4 := arithEq_of_rows h (i := 4) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_4
  simp only [← c3621, ← c3622, ← c3623] at e_93_4
  have e_93_5 := arithEq_of_rows h (i := 5) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_5
  simp only [← c3624, ← c3625, ← c3626] at e_93_5
  have hr := e_93_5
  simp only [e_93_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f685 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 31) = bselect (a (.wire 86 11)) (a (.wire 96 14)) (a (.wire 80 35)) := by
  have c3627 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3628 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3629 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3630 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3631 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3632 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_6 := arithEq_of_rows h (i := 6) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_6
  simp only [← c3627, ← c3628, ← c3629] at e_93_6
  have e_93_7 := arithEq_of_rows h (i := 7) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_7
  simp only [← c3630, ← c3631, ← c3632] at e_93_7
  have hr := e_93_7
  simp only [e_93_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f686 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 39) = bselect (a (.wire 86 11)) (a (.wire 96 15)) (a (.wire 80 43)) := by
  have c3633 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3634 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3635 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3636 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3637 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3638 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_8 := arithEq_of_rows h (i := 8) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_8
  simp only [← c3633, ← c3634, ← c3635] at e_93_8
  have e_93_9 := arithEq_of_rows h (i := 9) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_9
  simp only [← c3636, ← c3637, ← c3638] at e_93_9
  have hr := e_93_9
  simp only [e_93_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f687 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 97 1), a (.wire 97 2), a (.wire 97 3), a (.wire 97 4), a (.wire 97 5)] := by
  have c3639 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3640 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3641 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3642 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3643 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3644 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3645 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3646 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3647 := (leafCircuit_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3648 := (leafCircuit_copies114 a h).1
  have c3649 := (leafCircuit_copies114 a h).2.1
  have c3650 := (leafCircuit_copies114 a h).2.2.1
  have c3651 := (leafCircuit_copies114 a h).2.2.2.1
  have c3652 := (leafCircuit_copies114 a h).2.2.2.2.1
  have c3653 := (leafCircuit_copies114 a h).2.2.2.2.2.1
  have c3654 := (leafCircuit_copies114 a h).2.2.2.2.2.2.1
  have c3655 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.1
  have c3656 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.1
  have c3657 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.1
  have c3658 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3659 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3660 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3661 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3662 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3663 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3664 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3665 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3666 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3667 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3668 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3669 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3670 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3671 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3672 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3673 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3674 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3675 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3676 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3677 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3678 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3679 := (leafCircuit_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3680 := (leafCircuit_copies115 a h).1
  have c3681 := (leafCircuit_copies115 a h).2.1
  have c3682 := (leafCircuit_copies115 a h).2.2.1
  have c3683 := (leafCircuit_copies115 a h).2.2.2.1
  have c3684 := (leafCircuit_copies115 a h).2.2.2.2.1
  have c3685 := (leafCircuit_copies115 a h).2.2.2.2.2.1
  have c3686 := (leafCircuit_copies115 a h).2.2.2.2.2.2.1
  have c3687 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.1
  have c3688 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.1
  have c3689 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.1
  have c3690 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3691 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3692 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3693 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 97) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c3639.trans k0
    · exact c3640.trans k0
    · exact c3641.trans k0
    · exact c3642.trans k0
    · exact c3643.trans k0
    · exact c3644.trans k0
    · exact c3645.trans k0
    · exact c3646.trans k0
    · exact c3647.trans k0
    · exact c3648.trans k0
    · exact c3649.trans k0
    · exact c3650.trans k0
    · exact c3651.trans k0
    · exact c3652.trans k0
    · exact c3653.trans k0
    · exact c3654.trans k0
    · exact c3655.trans k0
    · exact c3656.trans k0
    · exact c3657.trans k0
    · exact c3658.trans k0
    · exact c3659.trans k0
    · exact c3660.trans k0
    · exact c3661.trans k0
    · exact c3662.trans k0
    · exact c3663.trans k0
    · exact c3664.trans k0
    · exact c3665.trans k0
    · exact c3666.trans k0
    · exact c3667.trans k0
    · exact c3668.trans k0
    · exact c3669.trans k0
    · exact c3670.trans k0
    · exact c3671.trans k0
    · exact c3672.trans k0
    · exact c3673.trans k0
    · exact c3674.trans k0
    · exact c3675.trans k0
    · exact c3676.trans k0
    · exact c3677.trans k0
    · exact c3678.trans k0
    · exact c3679.trans k0
    · exact c3680.trans k0
    · exact c3681.trans k0
    · exact c3682.trans k0
    · exact c3683.trans k0
    · exact c3684.trans k0
    · exact c3685.trans k0
    · exact c3686.trans k0
    · exact c3687.trans k0
    · exact c3688.trans k0
    · exact c3689.trans k0
    · exact c3690.trans k0
    · exact c3691.trans k0
    · exact c3692.trans k0
    )
  rwa [c3693] at hr

theorem leafCircuit_f688 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f689 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 5) = band (a (.virt 292)) (a (.wire 97 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f690 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 5) = band (a (.wire 97 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f691 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 5) = bor (a (.virt 14)) (a (.wire 97 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f692 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 97 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f693 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f694 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 5) = a (.virt 14) + a (.wire 97 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f695 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 5) = a (.wire 97 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f696 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 43) = bnot (a (.wire 97 5)) := by
  have c3694 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3695 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3696 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_93_10 := arithEq_of_rows h (i := 10) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_10
  simp only [← c3694, k3, ← c3695, k3, ← c3696] at e_93_10
  have hr := e_93_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f697 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 43) = band (a (.virt 292)) (a (.wire 93 43)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f698 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f699 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 4) = band (a (.virt 292)) (a (.wire 97 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f700 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 7) = band (a (.wire 97 4)) (a (.wire 93 43)) := by
  have c3697 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3698 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3699 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_1 := arithEq_of_rows h (i := 1) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_1
  simp only [← c3697, ← c3698, ← c3699] at e_89_1
  have hr := e_89_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f701 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 55) = bor (a (.wire 97 5)) (a (.wire 89 7)) := by
  have c3700 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3701 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3702 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3703 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3704 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3705 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_42_14 := arithEq_of_rows h (i := 14) leafCircuit_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_14
  simp only [← c3700, ← c3701, ← c3702] at e_42_14
  have e_86_13 := arithEq_of_rows h (i := 13) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_13
  simp only [← c3703, ← c3704, k3, ← c3705] at e_86_13
  have hr := e_86_13
  simp only [e_42_14] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f702 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 97 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f703 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f704 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 4) = a (.virt 14) + a (.wire 97 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f705 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 4) = a (.wire 97 4) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f706 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 47) = bnot (a (.wire 97 4)) := by
  have c3706 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3707 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3708 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_93_11 := arithEq_of_rows h (i := 11) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_11
  simp only [← c3706, k3, ← c3707, k3, ← c3708] at e_93_11
  have hr := e_93_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f707 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 11) = band (a (.wire 93 43)) (a (.wire 93 47)) := by
  have c3709 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3710 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3711 := (leafCircuit_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_89_2 := arithEq_of_rows h (i := 2) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_2
  simp only [← c3709, ← c3710, ← c3711] at e_89_2
  have hr := e_89_2
  simp only [band]
  linear_combination hr

theorem leafCircuit_f708 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f709 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 97 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f710 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 89 11)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f711 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 55) = bor (a (.wire 86 55)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f712 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 3) = a (.virt 292) * a (.wire 97 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f713 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 15) = a (.virt 294) * a (.wire 97 3) := by
  have c3712 := (leafCircuit_copies116 a h).1
  have c3713 := (leafCircuit_copies116 a h).2.1
  have c3714 := (leafCircuit_copies116 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_89_3 := arithEq_of_rows h (i := 3) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_3
  simp only [← c3712, k5, ← c3713, ← c3714, k5] at e_89_3
  have hr := e_89_3
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f714 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 59) = a (.virt 292) + a (.wire 97 3) := by
  have c3715 := (leafCircuit_copies116 a h).2.2.2.1
  have c3716 := (leafCircuit_copies116 a h).2.2.2.2.1
  have c3717 := (leafCircuit_copies116 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_86_14 := arithEq_of_rows h (i := 14) leafCircuit_row86 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_14
  simp only [← c3715, k3, ← c3716, k3, ← c3717] at e_86_14
  have hr := e_86_14
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f715 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 51) = a (.wire 86 59) - a (.wire 89 15) := by
  have c3718 := (leafCircuit_copies116 a h).2.2.2.2.2.2.1
  have c3719 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.1
  have c3720 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_93_12 := arithEq_of_rows h (i := 12) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_12
  simp only [← c3718, ← c3719, k3, ← c3720] at e_93_12
  have hr := e_93_12
  linear_combination hr

theorem leafCircuit_f716 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 55) = bnot (a (.wire 93 51)) := by
  have c3721 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.1
  have c3722 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3723 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_93_13 := arithEq_of_rows h (i := 13) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_13
  simp only [← c3721, k3, ← c3722, k3, ← c3723] at e_93_13
  have hr := e_93_13
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f717 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 19) = band (a (.wire 89 11)) (a (.wire 93 55)) := by
  have c3724 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3725 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3726 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_4 := arithEq_of_rows h (i := 4) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_4
  simp only [← c3724, ← c3725, ← c3726] at e_89_4
  have hr := e_89_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f718 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f719 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 97 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f720 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 89 19)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f721 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 55) = bor (a (.wire 86 55)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f722 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 2) = a (.virt 292) * a (.wire 97 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f723 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 23) = a (.virt 294) * a (.wire 97 2) := by
  have c3727 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3728 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3729 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_89_5 := arithEq_of_rows h (i := 5) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_5
  simp only [← c3727, k5, ← c3728, ← c3729, k5] at e_89_5
  have hr := e_89_5
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f724 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 3) = a (.virt 292) + a (.wire 97 2) := by
  have c3730 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3731 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3732 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_0 := arithEq_of_rows h (i := 0) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_0
  simp only [← c3730, k3, ← c3731, k3, ← c3732] at e_98_0
  have hr := e_98_0
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f725 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 93 59) = a (.wire 98 3) - a (.wire 89 23) := by
  have c3733 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3734 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3735 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_93_14 := arithEq_of_rows h (i := 14) leafCircuit_row93 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_14
  simp only [← c3733, ← c3734, k3, ← c3735] at e_93_14
  have hr := e_93_14
  linear_combination hr

theorem leafCircuit_f726 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 99 3) = bnot (a (.wire 93 59)) := by
  have c3736 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3737 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3738 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_99_0 := arithEq_of_rows h (i := 0) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_0
  simp only [← c3736, k3, ← c3737, k3, ← c3738] at e_99_0
  have hr := e_99_0
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f727 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 27) = band (a (.wire 89 19)) (a (.wire 99 3)) := by
  have c3739 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3740 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3741 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_6 := arithEq_of_rows h (i := 6) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_6
  simp only [← c3739, ← c3740, ← c3741] at e_89_6
  have hr := e_89_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f728 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f729 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 97 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f730 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 89 27)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f731 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 86 55) = bor (a (.wire 86 55)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f732 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 97 1) = a (.virt 292) * a (.wire 97 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f733 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 31) = a (.virt 294) * a (.wire 97 1) := by
  have c3742 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3743 := (leafCircuit_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3744 := (leafCircuit_copies117 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_89_7 := arithEq_of_rows h (i := 7) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_7
  simp only [← c3742, k5, ← c3743, ← c3744, k5] at e_89_7
  have hr := e_89_7
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f734 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 7) = a (.virt 292) + a (.wire 97 1) := by
  have c3745 := (leafCircuit_copies117 a h).2.1
  have c3746 := (leafCircuit_copies117 a h).2.2.1
  have c3747 := (leafCircuit_copies117 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_1 := arithEq_of_rows h (i := 1) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_1
  simp only [← c3745, k3, ← c3746, k3, ← c3747] at e_98_1
  have hr := e_98_1
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f735 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 99 7) = a (.wire 98 7) - a (.wire 89 31) := by
  have c3748 := (leafCircuit_copies117 a h).2.2.2.2.1
  have c3749 := (leafCircuit_copies117 a h).2.2.2.2.2.1
  have c3750 := (leafCircuit_copies117 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_99_1 := arithEq_of_rows h (i := 1) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_1
  simp only [← c3748, ← c3749, k3, ← c3750] at e_99_1
  have hr := e_99_1
  linear_combination hr

theorem leafCircuit_f736 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 99 11) = bnot (a (.wire 99 7)) := by
  have c3751 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.1
  have c3752 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.1
  have c3753 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_99_2 := arithEq_of_rows h (i := 2) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_2
  simp only [← c3751, k3, ← c3752, k3, ← c3753] at e_99_2
  have hr := e_99_2
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f737 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 89 35) = band (a (.wire 89 27)) (a (.wire 99 11)) := by
  have c3754 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3755 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3756 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_8 := arithEq_of_rows h (i := 8) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_8
  simp only [← c3754, ← c3755, ← c3756] at e_89_8
  have hr := e_89_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f738 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 214)) 2 := by
  have c3757 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3758 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3759 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3760 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3761 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3762 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3763 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3764 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3765 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3766 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3767 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3768 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3769 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3770 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3771 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3772 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3773 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3774 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3775 := (leafCircuit_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3776 := (leafCircuit_copies118 a h).1
  have c3777 := (leafCircuit_copies118 a h).2.1
  have c3778 := (leafCircuit_copies118 a h).2.2.1
  have c3779 := (leafCircuit_copies118 a h).2.2.2.1
  have c3780 := (leafCircuit_copies118 a h).2.2.2.2.1
  have c3781 := (leafCircuit_copies118 a h).2.2.2.2.2.1
  have c3782 := (leafCircuit_copies118 a h).2.2.2.2.2.2.1
  have c3783 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.1
  have c3784 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.1
  have c3785 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.1
  have c3786 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3787 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3788 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3789 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3790 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3791 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3792 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3793 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3794 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3795 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3796 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3797 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3798 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3799 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3800 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3801 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3802 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3803 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3804 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3805 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3806 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3807 := (leafCircuit_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3808 := (leafCircuit_copies119 a h).1
  have c3809 := (leafCircuit_copies119 a h).2.1
  have c3810 := (leafCircuit_copies119 a h).2.2.1
  have c3811 := (leafCircuit_copies119 a h).2.2.2.1
  have c3812 := (leafCircuit_copies119 a h).2.2.2.2.1
  have c3813 := (leafCircuit_copies119 a h).2.2.2.2.2.1
  have c3814 := (leafCircuit_copies119 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 100) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c3757.trans k0
    · exact c3758.trans k0
    · exact c3759.trans k0
    · exact c3760.trans k0
    · exact c3761.trans k0
    · exact c3762.trans k0
    · exact c3763.trans k0
    · exact c3764.trans k0
    · exact c3765.trans k0
    · exact c3766.trans k0
    · exact c3767.trans k0
    · exact c3768.trans k0
    · exact c3769.trans k0
    · exact c3770.trans k0
    · exact c3771.trans k0
    · exact c3772.trans k0
    · exact c3773.trans k0
    · exact c3774.trans k0
    · exact c3775.trans k0
    · exact c3776.trans k0
    · exact c3777.trans k0
    · exact c3778.trans k0
    · exact c3779.trans k0
    · exact c3780.trans k0
    · exact c3781.trans k0
    · exact c3782.trans k0
    · exact c3783.trans k0
    · exact c3784.trans k0
    · exact c3785.trans k0
    · exact c3786.trans k0
    · exact c3787.trans k0
    · exact c3788.trans k0
    · exact c3789.trans k0
    · exact c3790.trans k0
    · exact c3791.trans k0
    · exact c3792.trans k0
    · exact c3793.trans k0
    · exact c3794.trans k0
    · exact c3795.trans k0
    · exact c3796.trans k0
    · exact c3797.trans k0
    · exact c3798.trans k0
    · exact c3799.trans k0
    · exact c3800.trans k0
    · exact c3801.trans k0
    · exact c3802.trans k0
    · exact c3803.trans k0
    · exact c3804.trans k0
    · exact c3805.trans k0
    · exact c3806.trans k0
    · exact c3807.trans k0
    · exact c3808.trans k0
    · exact c3809.trans k0
    · exact c3810.trans k0
    · exact c3811.trans k0
    · exact c3812.trans k0
    · exact c3813.trans k0
    )
  rwa [c3814] at hr

theorem leafCircuit_f739 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 214)) (a (.virt 14)) (a (.virt 352)) (a (.virt 353)) := by
  have c3815 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.1
  have c3816 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.1
  have c3817 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.1
  have c3818 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3819 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3820 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3821 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3822 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3823 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3824 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3825 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3826 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3827 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3828 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_89_9 := arithEq_of_rows h (i := 9) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_9
  simp only [← c3818, ← c3819, ← c3820] at e_89_9
  have e_89_10 := arithEq_of_rows h (i := 10) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_10
  simp only [← c3821, ← c3822, ← c3823] at e_89_10
  have e_99_3 := arithEq_of_rows h (i := 3) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_3
  simp only [← c3815, k3, ← c3816, k3, ← c3817] at e_99_3
  have e_99_4 := arithEq_of_rows h (i := 4) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_4
  simp only [← c3824, ← c3825, k3, ← c3826] at e_99_4
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_89_9
    linear_combination c3827.trans k0 - hc
  · have hc := e_99_4
    simp only [e_99_3, e_89_10] at hc
    linear_combination c3828.trans k0 - hc

theorem leafCircuit_f740 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 214)) (a (.virt 292)) (a (.virt 354)) (a (.virt 355)) := by
  have c3829 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3830 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3831 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3832 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3833 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3834 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3835 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3836 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3837 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3838 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3839 := (leafCircuit_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3840 := (leafCircuit_copies120 a h).1
  have c3841 := (leafCircuit_copies120 a h).2.1
  have c3842 := (leafCircuit_copies120 a h).2.2.1
  have c3843 := (leafCircuit_copies120 a h).2.2.2.1
  have c3844 := (leafCircuit_copies120 a h).2.2.2.2.1
  have c3845 := (leafCircuit_copies120 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_89_11 := arithEq_of_rows h (i := 11) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_11
  simp only [← c3835, ← c3836, ← c3837] at e_89_11
  have e_89_12 := arithEq_of_rows h (i := 12) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_12
  simp only [← c3838, ← c3839, ← c3840] at e_89_12
  have e_99_5 := arithEq_of_rows h (i := 5) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_5
  simp only [← c3829, k3, ← c3830, k3, ← c3831] at e_99_5
  have e_99_6 := arithEq_of_rows h (i := 6) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_6
  simp only [← c3832, ← c3833, k3, ← c3834, k3] at e_99_6
  have e_99_7 := arithEq_of_rows h (i := 7) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_7
  simp only [← c3841, ← c3842, k3, ← c3843] at e_99_7
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_89_11
    simp only [e_99_6] at hc
    linear_combination c3844.trans k0 - hc
  · have hc := e_99_7
    simp only [e_99_5, e_89_12, e_99_6] at hc
    linear_combination c3845.trans k0 - hc

theorem leafCircuit_f741 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 214)) (a (.virt 294)) (a (.virt 356)) (a (.virt 357)) := by
  have c3846 := (leafCircuit_copies120 a h).2.2.2.2.2.2.1
  have c3847 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.1
  have c3848 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.1
  have c3849 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.1
  have c3850 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3851 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3852 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3853 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3854 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3855 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3856 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3857 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3858 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3859 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3860 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3861 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3862 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_89_13 := arithEq_of_rows h (i := 13) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_13
  simp only [← c3852, ← c3853, ← c3854] at e_89_13
  have e_89_14 := arithEq_of_rows h (i := 14) leafCircuit_row89 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_14
  simp only [← c3855, ← c3856, ← c3857] at e_89_14
  have e_99_8 := arithEq_of_rows h (i := 8) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_8
  simp only [← c3846, k3, ← c3847, k3, ← c3848] at e_99_8
  have e_99_9 := arithEq_of_rows h (i := 9) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_9
  simp only [← c3849, ← c3850, k3, ← c3851, k5] at e_99_9
  have e_99_10 := arithEq_of_rows h (i := 10) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_10
  simp only [← c3858, ← c3859, k3, ← c3860] at e_99_10
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_89_13
    simp only [e_99_9] at hc
    linear_combination c3861.trans k0 - hc
  · have hc := e_99_10
    simp only [e_99_8, e_89_14, e_99_9] at hc
    linear_combination c3862.trans k0 - hc

theorem leafCircuit_f742 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 214)) (a (.virt 295)) (a (.virt 358)) (a (.virt 359)) := by
  have c3863 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3864 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3865 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3866 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3867 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3868 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3869 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3870 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3871 := (leafCircuit_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3872 := (leafCircuit_copies121 a h).1
  have c3873 := (leafCircuit_copies121 a h).2.1
  have c3874 := (leafCircuit_copies121 a h).2.2.1
  have c3875 := (leafCircuit_copies121 a h).2.2.2.1
  have c3876 := (leafCircuit_copies121 a h).2.2.2.2.1
  have c3877 := (leafCircuit_copies121 a h).2.2.2.2.2.1
  have c3878 := (leafCircuit_copies121 a h).2.2.2.2.2.2.1
  have c3879 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_99_11 := arithEq_of_rows h (i := 11) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_11
  simp only [← c3863, k3, ← c3864, k3, ← c3865] at e_99_11
  have e_99_12 := arithEq_of_rows h (i := 12) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_12
  simp only [← c3866, ← c3867, k3, ← c3868, k6] at e_99_12
  have e_99_13 := arithEq_of_rows h (i := 13) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_13
  simp only [← c3875, ← c3876, k3, ← c3877] at e_99_13
  have e_101_0 := arithEq_of_rows h (i := 0) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_0
  simp only [← c3869, ← c3870, ← c3871] at e_101_0
  have e_101_1 := arithEq_of_rows h (i := 1) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_1
  simp only [← c3872, ← c3873, ← c3874] at e_101_1
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_101_0
    simp only [e_99_12] at hc
    linear_combination c3878.trans k0 - hc
  · have hc := e_99_13
    simp only [e_99_11, e_101_1, e_99_12] at hc
    linear_combination c3879.trans k0 - hc

theorem leafCircuit_f743 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 3) = bselect (a (.virt 352)) (a (.wire 93 15)) (a (.virt 99)) := by
  have c3880 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.1
  have c3881 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.1
  have c3882 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3883 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3884 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3885 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_14 := arithEq_of_rows h (i := 14) leafCircuit_row99 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_14
  simp only [← c3880, ← c3881, ← c3882] at e_99_14
  have e_102_0 := arithEq_of_rows h (i := 0) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_0
  simp only [← c3883, ← c3884, ← c3885] at e_102_0
  have hr := e_102_0
  simp only [e_99_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f744 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 11) = bselect (a (.virt 352)) (a (.wire 93 23)) (a (.virt 100)) := by
  have c3886 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3887 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3888 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3889 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3890 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3891 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_1 := arithEq_of_rows h (i := 1) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_1
  simp only [← c3886, ← c3887, ← c3888] at e_102_1
  have e_102_2 := arithEq_of_rows h (i := 2) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_2
  simp only [← c3889, ← c3890, ← c3891] at e_102_2
  have hr := e_102_2
  simp only [e_102_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f745 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 19) = bselect (a (.virt 352)) (a (.wire 93 31)) (a (.virt 101)) := by
  have c3892 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3893 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3894 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3895 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3896 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3897 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_3 := arithEq_of_rows h (i := 3) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_3
  simp only [← c3892, ← c3893, ← c3894] at e_102_3
  have e_102_4 := arithEq_of_rows h (i := 4) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_4
  simp only [← c3895, ← c3896, ← c3897] at e_102_4
  have hr := e_102_4
  simp only [e_102_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f746 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 27) = bselect (a (.virt 352)) (a (.wire 93 39)) (a (.virt 102)) := by
  have c3898 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3899 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3900 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3901 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3902 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3903 := (leafCircuit_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_102_5 := arithEq_of_rows h (i := 5) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_5
  simp only [← c3898, ← c3899, ← c3900] at e_102_5
  have e_102_6 := arithEq_of_rows h (i := 6) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_6
  simp only [← c3901, ← c3902, ← c3903] at e_102_6
  have hr := e_102_6
  simp only [e_102_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f747 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 35) = bselect (a (.virt 352)) (a (.virt 99)) (a (.virt 103)) := by
  have c3904 := (leafCircuit_copies122 a h).1
  have c3905 := (leafCircuit_copies122 a h).2.1
  have c3906 := (leafCircuit_copies122 a h).2.2.1
  have c3907 := (leafCircuit_copies122 a h).2.2.2.1
  have c3908 := (leafCircuit_copies122 a h).2.2.2.2.1
  have c3909 := (leafCircuit_copies122 a h).2.2.2.2.2.1
  have e_102_7 := arithEq_of_rows h (i := 7) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_7
  simp only [← c3904, ← c3905, ← c3906] at e_102_7
  have e_102_8 := arithEq_of_rows h (i := 8) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_8
  simp only [← c3907, ← c3908, ← c3909] at e_102_8
  have hr := e_102_8
  simp only [e_102_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f748 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 43) = bselect (a (.virt 354)) (a (.wire 93 15)) (a (.wire 102 35)) := by
  have c3910 := (leafCircuit_copies122 a h).2.2.2.2.2.2.1
  have c3911 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.1
  have c3912 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.1
  have c3913 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.1
  have c3914 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3915 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_9 := arithEq_of_rows h (i := 9) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_9
  simp only [← c3910, ← c3911, ← c3912] at e_102_9
  have e_102_10 := arithEq_of_rows h (i := 10) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_10
  simp only [← c3913, ← c3914, ← c3915] at e_102_10
  have hr := e_102_10
  simp only [e_102_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f749 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 51) = bselect (a (.virt 352)) (a (.virt 100)) (a (.virt 104)) := by
  have c3916 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3917 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3918 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3919 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3920 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3921 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_11 := arithEq_of_rows h (i := 11) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_11
  simp only [← c3916, ← c3917, ← c3918] at e_102_11
  have e_102_12 := arithEq_of_rows h (i := 12) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_12
  simp only [← c3919, ← c3920, ← c3921] at e_102_12
  have hr := e_102_12
  simp only [e_102_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f750 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 102 59) = bselect (a (.virt 354)) (a (.wire 93 23)) (a (.wire 102 51)) := by
  have c3922 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3923 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3924 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3925 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3926 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3927 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_13 := arithEq_of_rows h (i := 13) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_13
  simp only [← c3922, ← c3923, ← c3924] at e_102_13
  have e_102_14 := arithEq_of_rows h (i := 14) leafCircuit_row102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_14
  simp only [← c3925, ← c3926, ← c3927] at e_102_14
  have hr := e_102_14
  simp only [e_102_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f751 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 7) = bselect (a (.virt 352)) (a (.virt 101)) (a (.virt 105)) := by
  have c3928 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3929 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3930 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3931 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3932 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3933 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_0 := arithEq_of_rows h (i := 0) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_0
  simp only [← c3928, ← c3929, ← c3930] at e_103_0
  have e_103_1 := arithEq_of_rows h (i := 1) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_1
  simp only [← c3931, ← c3932, ← c3933] at e_103_1
  have hr := e_103_1
  simp only [e_103_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f752 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 15) = bselect (a (.virt 354)) (a (.wire 93 31)) (a (.wire 103 7)) := by
  have c3934 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3935 := (leafCircuit_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3936 := (leafCircuit_copies123 a h).1
  have c3937 := (leafCircuit_copies123 a h).2.1
  have c3938 := (leafCircuit_copies123 a h).2.2.1
  have c3939 := (leafCircuit_copies123 a h).2.2.2.1
  have e_103_2 := arithEq_of_rows h (i := 2) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_2
  simp only [← c3934, ← c3935, ← c3936] at e_103_2
  have e_103_3 := arithEq_of_rows h (i := 3) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_3
  simp only [← c3937, ← c3938, ← c3939] at e_103_3
  have hr := e_103_3
  simp only [e_103_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f753 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 23) = bselect (a (.virt 352)) (a (.virt 102)) (a (.virt 106)) := by
  have c3940 := (leafCircuit_copies123 a h).2.2.2.2.1
  have c3941 := (leafCircuit_copies123 a h).2.2.2.2.2.1
  have c3942 := (leafCircuit_copies123 a h).2.2.2.2.2.2.1
  have c3943 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.1
  have c3944 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.1
  have c3945 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.1
  have e_103_4 := arithEq_of_rows h (i := 4) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_4
  simp only [← c3940, ← c3941, ← c3942] at e_103_4
  have e_103_5 := arithEq_of_rows h (i := 5) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_5
  simp only [← c3943, ← c3944, ← c3945] at e_103_5
  have hr := e_103_5
  simp only [e_103_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f754 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 31) = bselect (a (.virt 354)) (a (.wire 93 39)) (a (.wire 103 23)) := by
  have c3946 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3947 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3948 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3949 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3950 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3951 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_6 := arithEq_of_rows h (i := 6) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_6
  simp only [← c3946, ← c3947, ← c3948] at e_103_6
  have e_103_7 := arithEq_of_rows h (i := 7) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_7
  simp only [← c3949, ← c3950, ← c3951] at e_103_7
  have hr := e_103_7
  simp only [e_103_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f755 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) := by
  have c3952 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3953 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3956 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_2 := arithEq_of_rows h (i := 2) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c3955, ← c3956, k3, ← c3957] at e_98_2
  have e_104_0 := arithEq_of_rows h (i := 0) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c3952, ← c3953, ← c3954] at e_104_0
  have hr := e_98_2
  simp only [e_104_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f756 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 39) = bselect (a (.wire 98 11)) (a (.virt 103)) (a (.virt 107)) := by
  have c3958 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3959 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3960 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3961 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3962 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3963 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_8 := arithEq_of_rows h (i := 8) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_8
  simp only [← c3958, ← c3959, ← c3960] at e_103_8
  have e_103_9 := arithEq_of_rows h (i := 9) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_9
  simp only [← c3961, ← c3962, ← c3963] at e_103_9
  have hr := e_103_9
  simp only [e_103_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f757 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 47) = bselect (a (.virt 356)) (a (.wire 93 15)) (a (.wire 103 39)) := by
  have c3964 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3965 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3966 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3967 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3968 := (leafCircuit_copies124 a h).1
  have c3969 := (leafCircuit_copies124 a h).2.1
  have e_103_10 := arithEq_of_rows h (i := 10) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_10
  simp only [← c3964, ← c3965, ← c3966] at e_103_10
  have e_103_11 := arithEq_of_rows h (i := 11) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_11
  simp only [← c3967, ← c3968, ← c3969] at e_103_11
  have hr := e_103_11
  simp only [e_103_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f758 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) := by
  have c3952 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3953 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3956 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_2 := arithEq_of_rows h (i := 2) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c3955, ← c3956, k3, ← c3957] at e_98_2
  have e_104_0 := arithEq_of_rows h (i := 0) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c3952, ← c3953, ← c3954] at e_104_0
  have hr := e_98_2
  simp only [e_104_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f759 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 103 55) = bselect (a (.wire 98 11)) (a (.virt 104)) (a (.virt 108)) := by
  have c3970 := (leafCircuit_copies124 a h).2.2.1
  have c3971 := (leafCircuit_copies124 a h).2.2.2.1
  have c3972 := (leafCircuit_copies124 a h).2.2.2.2.1
  have c3973 := (leafCircuit_copies124 a h).2.2.2.2.2.1
  have c3974 := (leafCircuit_copies124 a h).2.2.2.2.2.2.1
  have c3975 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.1
  have e_103_12 := arithEq_of_rows h (i := 12) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_12
  simp only [← c3970, ← c3971, ← c3972] at e_103_12
  have e_103_13 := arithEq_of_rows h (i := 13) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_13
  simp only [← c3973, ← c3974, ← c3975] at e_103_13
  have hr := e_103_13
  simp only [e_103_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f760 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 3) = bselect (a (.virt 356)) (a (.wire 93 23)) (a (.wire 103 55)) := by
  have c3976 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.1
  have c3977 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.1
  have c3978 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3979 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3980 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3981 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_14 := arithEq_of_rows h (i := 14) leafCircuit_row103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_14
  simp only [← c3976, ← c3977, ← c3978] at e_103_14
  have e_105_0 := arithEq_of_rows h (i := 0) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_0
  simp only [← c3979, ← c3980, ← c3981] at e_105_0
  have hr := e_105_0
  simp only [e_103_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f761 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) := by
  have c3952 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3953 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3956 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_2 := arithEq_of_rows h (i := 2) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c3955, ← c3956, k3, ← c3957] at e_98_2
  have e_104_0 := arithEq_of_rows h (i := 0) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c3952, ← c3953, ← c3954] at e_104_0
  have hr := e_98_2
  simp only [e_104_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f762 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 11) = bselect (a (.wire 98 11)) (a (.virt 105)) (a (.virt 109)) := by
  have c3982 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3983 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3984 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3985 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3986 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3987 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_1 := arithEq_of_rows h (i := 1) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_1
  simp only [← c3982, ← c3983, ← c3984] at e_105_1
  have e_105_2 := arithEq_of_rows h (i := 2) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_2
  simp only [← c3985, ← c3986, ← c3987] at e_105_2
  have hr := e_105_2
  simp only [e_105_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f763 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 19) = bselect (a (.virt 356)) (a (.wire 93 31)) (a (.wire 105 11)) := by
  have c3988 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3989 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3990 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3991 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3992 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3993 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_3 := arithEq_of_rows h (i := 3) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_3
  simp only [← c3988, ← c3989, ← c3990] at e_105_3
  have e_105_4 := arithEq_of_rows h (i := 4) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_4
  simp only [← c3991, ← c3992, ← c3993] at e_105_4
  have hr := e_105_4
  simp only [e_105_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f764 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) := by
  have c3952 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3953 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3956 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (leafCircuit_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_2 := arithEq_of_rows h (i := 2) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c3955, ← c3956, k3, ← c3957] at e_98_2
  have e_104_0 := arithEq_of_rows h (i := 0) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c3952, ← c3953, ← c3954] at e_104_0
  have hr := e_98_2
  simp only [e_104_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f765 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 27) = bselect (a (.wire 98 11)) (a (.virt 106)) (a (.virt 110)) := by
  have c3994 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3995 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3996 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3997 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3998 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3999 := (leafCircuit_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_105_5 := arithEq_of_rows h (i := 5) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_5
  simp only [← c3994, ← c3995, ← c3996] at e_105_5
  have e_105_6 := arithEq_of_rows h (i := 6) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_6
  simp only [← c3997, ← c3998, ← c3999] at e_105_6
  have hr := e_105_6
  simp only [e_105_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f766 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 35) = bselect (a (.virt 356)) (a (.wire 93 39)) (a (.wire 105 27)) := by
  have c4000 := (leafCircuit_copies125 a h).1
  have c4001 := (leafCircuit_copies125 a h).2.1
  have c4002 := (leafCircuit_copies125 a h).2.2.1
  have c4003 := (leafCircuit_copies125 a h).2.2.2.1
  have c4004 := (leafCircuit_copies125 a h).2.2.2.2.1
  have c4005 := (leafCircuit_copies125 a h).2.2.2.2.2.1
  have e_105_7 := arithEq_of_rows h (i := 7) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_7
  simp only [← c4000, ← c4001, ← c4002] at e_105_7
  have e_105_8 := arithEq_of_rows h (i := 8) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_8
  simp only [← c4003, ← c4004, ← c4005] at e_105_8
  have hr := e_105_8
  simp only [e_105_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f767 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 43) = bselect (a (.virt 358)) (a (.wire 93 15)) (a (.virt 107)) := by
  have c4006 := (leafCircuit_copies125 a h).2.2.2.2.2.2.1
  have c4007 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.1
  have c4008 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.1
  have c4009 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.1
  have c4010 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4011 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_9 := arithEq_of_rows h (i := 9) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_9
  simp only [← c4006, ← c4007, ← c4008] at e_105_9
  have e_105_10 := arithEq_of_rows h (i := 10) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_10
  simp only [← c4009, ← c4010, ← c4011] at e_105_10
  have hr := e_105_10
  simp only [e_105_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f768 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 51) = bselect (a (.virt 358)) (a (.wire 93 23)) (a (.virt 108)) := by
  have c4012 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4013 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4014 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4015 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4016 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4017 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_11 := arithEq_of_rows h (i := 11) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_11
  simp only [← c4012, ← c4013, ← c4014] at e_105_11
  have e_105_12 := arithEq_of_rows h (i := 12) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_12
  simp only [← c4015, ← c4016, ← c4017] at e_105_12
  have hr := e_105_12
  simp only [e_105_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f769 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 105 59) = bselect (a (.virt 358)) (a (.wire 93 31)) (a (.virt 109)) := by
  have c4018 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4019 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4020 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4021 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4022 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4023 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_13 := arithEq_of_rows h (i := 13) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_13
  simp only [← c4018, ← c4019, ← c4020] at e_105_13
  have e_105_14 := arithEq_of_rows h (i := 14) leafCircuit_row105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_14
  simp only [← c4021, ← c4022, ← c4023] at e_105_14
  have hr := e_105_14
  simp only [e_105_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f770 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 7) = bselect (a (.virt 358)) (a (.wire 93 39)) (a (.virt 110)) := by
  have c4024 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4025 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4026 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4027 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4028 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4029 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_0 := arithEq_of_rows h (i := 0) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_0
  simp only [← c4024, ← c4025, ← c4026] at e_106_0
  have e_106_1 := arithEq_of_rows h (i := 1) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_1
  simp only [← c4027, ← c4028, ← c4029] at e_106_1
  have hr := e_106_1
  simp only [e_106_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f771 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 109 12) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 0 ∧ a (.wire 109 13) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 1 ∧ a (.wire 109 14) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 2 ∧ a (.wire 109 15) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 3) := by
  have c4030 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4031 := (leafCircuit_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4032 := (leafCircuit_copies126 a h).1
  have c4033 := (leafCircuit_copies126 a h).2.1
  have c4034 := (leafCircuit_copies126 a h).2.2.1
  have c4035 := (leafCircuit_copies126 a h).2.2.2.1
  have c4036 := (leafCircuit_copies126 a h).2.2.2.2.1
  have c4037 := (leafCircuit_copies126 a h).2.2.2.2.2.1
  have c4038 := (leafCircuit_copies126 a h).2.2.2.2.2.2.1
  have c4039 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.1
  have c4040 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.1
  have c4041 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.1
  have c4042 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4043 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4044 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4045 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4046 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4047 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4048 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4049 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4050 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4051 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4052 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4053 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4054 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4055 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4056 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4057 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4058 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4059 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4060 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4061 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4062 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4063 := (leafCircuit_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4064 := (leafCircuit_copies127 a h).1
  have c4065 := (leafCircuit_copies127 a h).2.1
  have c4066 := (leafCircuit_copies127 a h).2.2.1
  have c4067 := (leafCircuit_copies127 a h).2.2.2.1
  have c4068 := (leafCircuit_copies127 a h).2.2.2.2.1
  have c4069 := (leafCircuit_copies127 a h).2.2.2.2.2.1
  have c4070 := (leafCircuit_copies127 a h).2.2.2.2.2.2.1
  have c4071 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.1
  have c4072 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.1
  have c4073 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.1
  have c4074 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4075 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4076 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4077 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4078 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4079 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4080 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4081 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4082 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4083 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4084 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4085 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4086 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4087 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4088 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4089 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4090 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4091 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4092 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_3 := arithEq_of_rows h (i := 3) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_3
  simp only [← c4042, ← c4043, k3, ← c4044] at e_98_3
  have e_98_4 := arithEq_of_rows h (i := 4) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_4
  simp only [← c4045, ← c4046, k3, ← c4047] at e_98_4
  have e_98_5 := arithEq_of_rows h (i := 5) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_5
  simp only [← c4048, ← c4049, k3, ← c4050] at e_98_5
  have e_98_6 := arithEq_of_rows h (i := 6) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_6
  simp only [← c4051, ← c4052, k3, ← c4053] at e_98_6
  have e_98_7 := arithEq_of_rows h (i := 7) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_7
  simp only [← c4054, ← c4055, k3, ← c4056] at e_98_7
  have e_98_8 := arithEq_of_rows h (i := 8) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_8
  simp only [← c4057, ← c4058, k3, ← c4059] at e_98_8
  have e_98_9 := arithEq_of_rows h (i := 9) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_9
  simp only [← c4060, ← c4061, k3, ← c4062] at e_98_9
  have e_98_10 := arithEq_of_rows h (i := 10) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_10
  simp only [← c4063, ← c4064, k3, ← c4065] at e_98_10
  have e_98_11 := arithEq_of_rows h (i := 11) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_11
  simp only [← c4078, ← c4079, k3, ← c4080, k3] at e_98_11
  have hin0 : poseidon2In a 107 = addBlock (fun _ => 0) [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31)] :=
    poseidon2In_first c4030.symm c4031.symm c4032.symm c4033.symm c4034.symm c4035.symm c4036.symm c4037.symm (c4038.symm.trans k0) (c4039.symm.trans k0) (c4040.symm.trans k0) (c4041.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 107) rfl rfl hin0
  have hin1 : poseidon2In a 108 = addBlock (poseidon2Out a 107) [a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] :=
    poseidon2In_chain (by linear_combination -c4066 + e_98_3) (by linear_combination -c4067 + e_98_4) (by linear_combination -c4068 + e_98_5) (by linear_combination -c4069 + e_98_6) (by linear_combination -c4070 + e_98_7) (by linear_combination -c4071 + e_98_8) (by linear_combination -c4072 + e_98_9) (by linear_combination -c4073 + e_98_10) c4074.symm c4075.symm c4076.symm c4077.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 108) rfl rfl hin1
  have hin2 : poseidon2In a 109 = addBlock (poseidon2Out a 108) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c4081 + e_98_11) (c4082.symm.trans (add_zero _).symm) (c4083.symm.trans (add_zero _).symm) (c4084.symm.trans (add_zero _).symm) (c4085.symm.trans (add_zero _).symm) (c4086.symm.trans (add_zero _).symm) (c4087.symm.trans (add_zero _).symm) (c4088.symm.trans (add_zero _).symm) c4089.symm c4090.symm c4091.symm c4092.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 109) rfl rfl hin2
  have hpad : pad10 [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] =
      [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31)] ++ ([a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f772 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 15) = bselect (a (.wire 86 55)) (a (.wire 109 12)) (a (.wire 93 15)) := by
  have c4093 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4094 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4095 := (leafCircuit_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4096 := (leafCircuit_copies128 a h).1
  have c4097 := (leafCircuit_copies128 a h).2.1
  have c4098 := (leafCircuit_copies128 a h).2.2.1
  have e_106_2 := arithEq_of_rows h (i := 2) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_2
  simp only [← c4093, ← c4094, ← c4095] at e_106_2
  have e_106_3 := arithEq_of_rows h (i := 3) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_3
  simp only [← c4096, ← c4097, ← c4098] at e_106_3
  have hr := e_106_3
  simp only [e_106_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f773 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 23) = bselect (a (.wire 86 55)) (a (.wire 109 13)) (a (.wire 93 23)) := by
  have c4099 := (leafCircuit_copies128 a h).2.2.2.1
  have c4100 := (leafCircuit_copies128 a h).2.2.2.2.1
  have c4101 := (leafCircuit_copies128 a h).2.2.2.2.2.1
  have c4102 := (leafCircuit_copies128 a h).2.2.2.2.2.2.1
  have c4103 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.1
  have c4104 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.1
  have e_106_4 := arithEq_of_rows h (i := 4) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_4
  simp only [← c4099, ← c4100, ← c4101] at e_106_4
  have e_106_5 := arithEq_of_rows h (i := 5) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_5
  simp only [← c4102, ← c4103, ← c4104] at e_106_5
  have hr := e_106_5
  simp only [e_106_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f774 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 31) = bselect (a (.wire 86 55)) (a (.wire 109 14)) (a (.wire 93 31)) := by
  have c4105 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.1
  have c4106 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4107 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4108 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4109 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4110 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_6 := arithEq_of_rows h (i := 6) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_6
  simp only [← c4105, ← c4106, ← c4107] at e_106_6
  have e_106_7 := arithEq_of_rows h (i := 7) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_7
  simp only [← c4108, ← c4109, ← c4110] at e_106_7
  have hr := e_106_7
  simp only [e_106_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f775 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 39) = bselect (a (.wire 86 55)) (a (.wire 109 15)) (a (.wire 93 39)) := by
  have c4111 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4112 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4113 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4114 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4115 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4116 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_8 := arithEq_of_rows h (i := 8) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_8
  simp only [← c4111, ← c4112, ← c4113] at e_106_8
  have e_106_9 := arithEq_of_rows h (i := 9) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_9
  simp only [← c4114, ← c4115, ← c4116] at e_106_9
  have hr := e_106_9
  simp only [e_106_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f776 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 110 1), a (.wire 110 2), a (.wire 110 3), a (.wire 110 4), a (.wire 110 5)] := by
  have c4117 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4118 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4119 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4120 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4121 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4122 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4123 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4124 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4125 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4126 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4127 := (leafCircuit_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4128 := (leafCircuit_copies129 a h).1
  have c4129 := (leafCircuit_copies129 a h).2.1
  have c4130 := (leafCircuit_copies129 a h).2.2.1
  have c4131 := (leafCircuit_copies129 a h).2.2.2.1
  have c4132 := (leafCircuit_copies129 a h).2.2.2.2.1
  have c4133 := (leafCircuit_copies129 a h).2.2.2.2.2.1
  have c4134 := (leafCircuit_copies129 a h).2.2.2.2.2.2.1
  have c4135 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.1
  have c4136 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.1
  have c4137 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.1
  have c4138 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4139 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4140 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4141 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4142 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4143 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4144 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4145 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4146 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4147 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4148 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4149 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4150 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4151 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4152 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4153 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4154 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4155 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4156 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4157 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4158 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4159 := (leafCircuit_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4160 := (leafCircuit_copies130 a h).1
  have c4161 := (leafCircuit_copies130 a h).2.1
  have c4162 := (leafCircuit_copies130 a h).2.2.1
  have c4163 := (leafCircuit_copies130 a h).2.2.2.1
  have c4164 := (leafCircuit_copies130 a h).2.2.2.2.1
  have c4165 := (leafCircuit_copies130 a h).2.2.2.2.2.1
  have c4166 := (leafCircuit_copies130 a h).2.2.2.2.2.2.1
  have c4167 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.1
  have c4168 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.1
  have c4169 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.1
  have c4170 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4171 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 110) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c4117.trans k0
    · exact c4118.trans k0
    · exact c4119.trans k0
    · exact c4120.trans k0
    · exact c4121.trans k0
    · exact c4122.trans k0
    · exact c4123.trans k0
    · exact c4124.trans k0
    · exact c4125.trans k0
    · exact c4126.trans k0
    · exact c4127.trans k0
    · exact c4128.trans k0
    · exact c4129.trans k0
    · exact c4130.trans k0
    · exact c4131.trans k0
    · exact c4132.trans k0
    · exact c4133.trans k0
    · exact c4134.trans k0
    · exact c4135.trans k0
    · exact c4136.trans k0
    · exact c4137.trans k0
    · exact c4138.trans k0
    · exact c4139.trans k0
    · exact c4140.trans k0
    · exact c4141.trans k0
    · exact c4142.trans k0
    · exact c4143.trans k0
    · exact c4144.trans k0
    · exact c4145.trans k0
    · exact c4146.trans k0
    · exact c4147.trans k0
    · exact c4148.trans k0
    · exact c4149.trans k0
    · exact c4150.trans k0
    · exact c4151.trans k0
    · exact c4152.trans k0
    · exact c4153.trans k0
    · exact c4154.trans k0
    · exact c4155.trans k0
    · exact c4156.trans k0
    · exact c4157.trans k0
    · exact c4158.trans k0
    · exact c4159.trans k0
    · exact c4160.trans k0
    · exact c4161.trans k0
    · exact c4162.trans k0
    · exact c4163.trans k0
    · exact c4164.trans k0
    · exact c4165.trans k0
    · exact c4166.trans k0
    · exact c4167.trans k0
    · exact c4168.trans k0
    · exact c4169.trans k0
    · exact c4170.trans k0
    )
  rwa [c4171] at hr

theorem leafCircuit_f777 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f778 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 5) = band (a (.virt 292)) (a (.wire 110 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f779 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 5) = band (a (.wire 110 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f780 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 5) = bor (a (.virt 14)) (a (.wire 110 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f781 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 110 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f782 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f783 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 5) = a (.virt 14) + a (.wire 110 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f784 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 5) = a (.wire 110 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f785 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 43) = bnot (a (.wire 110 5)) := by
  have c4172 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4173 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4174 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_106_10 := arithEq_of_rows h (i := 10) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_10
  simp only [← c4172, k3, ← c4173, k3, ← c4174] at e_106_10
  have hr := e_106_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f786 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 43) = band (a (.virt 292)) (a (.wire 106 43)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f787 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f788 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 110 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f789 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 106 43)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f790 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 5) = bor (a (.wire 110 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f791 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 4) = a (.virt 292) * a (.wire 110 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f792 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 11) = a (.virt 294) * a (.wire 110 4) := by
  have c4175 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4176 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4177 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_101_2 := arithEq_of_rows h (i := 2) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_2
  simp only [← c4175, k5, ← c4176, ← c4177, k5] at e_101_2
  have hr := e_101_2
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f793 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 51) = a (.virt 292) + a (.wire 110 4) := by
  have c4178 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4179 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4180 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_12 := arithEq_of_rows h (i := 12) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_12
  simp only [← c4178, k3, ← c4179, k3, ← c4180] at e_98_12
  have hr := e_98_12
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f794 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 47) = a (.wire 98 51) - a (.wire 101 11) := by
  have c4181 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4182 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4183 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_106_11 := arithEq_of_rows h (i := 11) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_11
  simp only [← c4181, ← c4182, k3, ← c4183] at e_106_11
  have hr := e_106_11
  linear_combination hr

theorem leafCircuit_f795 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 51) = bnot (a (.wire 106 47)) := by
  have c4184 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4185 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4186 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_106_12 := arithEq_of_rows h (i := 12) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_12
  simp only [← c4184, k3, ← c4185, k3, ← c4186] at e_106_12
  have hr := e_106_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f796 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 15) = band (a (.wire 106 43)) (a (.wire 106 51)) := by
  have c4187 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4188 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4189 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_3 := arithEq_of_rows h (i := 3) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_3
  simp only [← c4187, ← c4188, ← c4189] at e_101_3
  have hr := e_101_3
  simp only [band]
  linear_combination hr

theorem leafCircuit_f797 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f798 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 3) = band (a (.virt 292)) (a (.wire 110 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f799 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 19) = band (a (.wire 110 3)) (a (.wire 101 15)) := by
  have c4190 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4191 := (leafCircuit_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4192 := (leafCircuit_copies131 a h).1
  have e_101_4 := arithEq_of_rows h (i := 4) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_4
  simp only [← c4190, ← c4191, ← c4192] at e_101_4
  have hr := e_101_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f800 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 55) = bor (a (.wire 110 5)) (a (.wire 101 19)) := by
  have c4193 := (leafCircuit_copies131 a h).2.1
  have c4194 := (leafCircuit_copies131 a h).2.2.1
  have c4195 := (leafCircuit_copies131 a h).2.2.2.1
  have c4196 := (leafCircuit_copies131 a h).2.2.2.2.1
  have c4197 := (leafCircuit_copies131 a h).2.2.2.2.2.1
  have c4198 := (leafCircuit_copies131 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_13 := arithEq_of_rows h (i := 13) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_13
  simp only [← c4196, ← c4197, k3, ← c4198] at e_98_13
  have e_104_1 := arithEq_of_rows h (i := 1) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_1
  simp only [← c4193, ← c4194, ← c4195] at e_104_1
  have hr := e_98_13
  simp only [e_104_1] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f801 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 110 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f802 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f803 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 3) = a (.virt 14) + a (.wire 110 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f804 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 3) = a (.wire 110 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f805 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 55) = bnot (a (.wire 110 3)) := by
  have c4199 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.1
  have c4200 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.1
  have c4201 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_106_13 := arithEq_of_rows h (i := 13) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_13
  simp only [← c4199, k3, ← c4200, k3, ← c4201] at e_106_13
  have hr := e_106_13
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f806 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 23) = band (a (.wire 101 15)) (a (.wire 106 55)) := by
  have c4202 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4203 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4204 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_5 := arithEq_of_rows h (i := 5) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_5
  simp only [← c4202, ← c4203, ← c4204] at e_101_5
  have hr := e_101_5
  simp only [band]
  linear_combination hr

theorem leafCircuit_f807 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f808 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 2) = band (a (.virt 292)) (a (.wire 110 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f809 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 27) = band (a (.wire 110 2)) (a (.wire 101 23)) := by
  have c4205 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4206 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4207 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_6 := arithEq_of_rows h (i := 6) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_6
  simp only [← c4205, ← c4206, ← c4207] at e_101_6
  have hr := e_101_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f810 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 98 59) = bor (a (.wire 98 55)) (a (.wire 101 27)) := by
  have c4208 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4209 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4210 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4211 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4212 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4213 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_98_14 := arithEq_of_rows h (i := 14) leafCircuit_row98 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_14
  simp only [← c4211, ← c4212, k3, ← c4213] at e_98_14
  have e_104_2 := arithEq_of_rows h (i := 2) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_2
  simp only [← c4208, ← c4209, ← c4210] at e_104_2
  have hr := e_98_14
  simp only [e_104_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f811 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 110 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f812 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f813 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 2) = a (.virt 14) + a (.wire 110 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f814 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 2) = a (.wire 110 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f815 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 106 59) = bnot (a (.wire 110 2)) := by
  have c4214 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4215 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4216 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_106_14 := arithEq_of_rows h (i := 14) leafCircuit_row106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_14
  simp only [← c4214, k3, ← c4215, k3, ← c4216] at e_106_14
  have hr := e_106_14
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f816 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 31) = band (a (.wire 101 23)) (a (.wire 106 59)) := by
  have c4217 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4218 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4219 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_7 := arithEq_of_rows h (i := 7) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_7
  simp only [← c4217, ← c4218, ← c4219] at e_101_7
  have hr := e_101_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f817 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f818 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 1) = band (a (.virt 292)) (a (.wire 110 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f819 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 35) = band (a (.wire 110 1)) (a (.wire 101 31)) := by
  have c4220 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4221 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4222 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_8 := arithEq_of_rows h (i := 8) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_8
  simp only [← c4220, ← c4221, ← c4222] at e_101_8
  have hr := e_101_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f820 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 3) = bor (a (.wire 98 59)) (a (.wire 101 35)) := by
  have c4223 := (leafCircuit_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4224 := (leafCircuit_copies132 a h).1
  have c4225 := (leafCircuit_copies132 a h).2.1
  have c4226 := (leafCircuit_copies132 a h).2.2.1
  have c4227 := (leafCircuit_copies132 a h).2.2.2.1
  have c4228 := (leafCircuit_copies132 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_3 := arithEq_of_rows h (i := 3) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_3
  simp only [← c4223, ← c4224, ← c4225] at e_104_3
  have e_111_0 := arithEq_of_rows h (i := 0) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_0
  simp only [← c4226, ← c4227, k3, ← c4228] at e_111_0
  have hr := e_111_0
  simp only [e_104_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f821 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 110 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f822 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f823 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 1) = a (.virt 14) + a (.wire 110 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f824 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 110 1) = a (.wire 110 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f825 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 112 3) = bnot (a (.wire 110 1)) := by
  have c4229 := (leafCircuit_copies132 a h).2.2.2.2.2.1
  have c4230 := (leafCircuit_copies132 a h).2.2.2.2.2.2.1
  have c4231 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_112_0 := arithEq_of_rows h (i := 0) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_0
  simp only [← c4229, k3, ← c4230, k3, ← c4231] at e_112_0
  have hr := e_112_0
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f826 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 101 39) = band (a (.wire 101 31)) (a (.wire 112 3)) := by
  have c4232 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.1
  have c4233 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.1
  have c4234 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_101_9 := arithEq_of_rows h (i := 9) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_9
  simp only [← c4232, ← c4233, ← c4234] at e_101_9
  have hr := e_101_9
  simp only [band]
  linear_combination hr

theorem leafCircuit_f827 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 215)) 2 := by
  have c4235 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4236 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4237 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4238 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4239 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4240 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4241 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4242 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4243 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4244 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4245 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4246 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4247 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4248 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4249 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4250 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4251 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4252 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4253 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4254 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4255 := (leafCircuit_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4256 := (leafCircuit_copies133 a h).1
  have c4257 := (leafCircuit_copies133 a h).2.1
  have c4258 := (leafCircuit_copies133 a h).2.2.1
  have c4259 := (leafCircuit_copies133 a h).2.2.2.1
  have c4260 := (leafCircuit_copies133 a h).2.2.2.2.1
  have c4261 := (leafCircuit_copies133 a h).2.2.2.2.2.1
  have c4262 := (leafCircuit_copies133 a h).2.2.2.2.2.2.1
  have c4263 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.1
  have c4264 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.1
  have c4265 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.1
  have c4266 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4267 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4268 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4269 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4270 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4271 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4272 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4273 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4274 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4275 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4276 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4277 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4278 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4279 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4280 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4281 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4282 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4283 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4284 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4285 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4286 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4287 := (leafCircuit_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4288 := (leafCircuit_copies134 a h).1
  have c4289 := (leafCircuit_copies134 a h).2.1
  have c4290 := (leafCircuit_copies134 a h).2.2.1
  have c4291 := (leafCircuit_copies134 a h).2.2.2.1
  have c4292 := (leafCircuit_copies134 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 113) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c4235.trans k0
    · exact c4236.trans k0
    · exact c4237.trans k0
    · exact c4238.trans k0
    · exact c4239.trans k0
    · exact c4240.trans k0
    · exact c4241.trans k0
    · exact c4242.trans k0
    · exact c4243.trans k0
    · exact c4244.trans k0
    · exact c4245.trans k0
    · exact c4246.trans k0
    · exact c4247.trans k0
    · exact c4248.trans k0
    · exact c4249.trans k0
    · exact c4250.trans k0
    · exact c4251.trans k0
    · exact c4252.trans k0
    · exact c4253.trans k0
    · exact c4254.trans k0
    · exact c4255.trans k0
    · exact c4256.trans k0
    · exact c4257.trans k0
    · exact c4258.trans k0
    · exact c4259.trans k0
    · exact c4260.trans k0
    · exact c4261.trans k0
    · exact c4262.trans k0
    · exact c4263.trans k0
    · exact c4264.trans k0
    · exact c4265.trans k0
    · exact c4266.trans k0
    · exact c4267.trans k0
    · exact c4268.trans k0
    · exact c4269.trans k0
    · exact c4270.trans k0
    · exact c4271.trans k0
    · exact c4272.trans k0
    · exact c4273.trans k0
    · exact c4274.trans k0
    · exact c4275.trans k0
    · exact c4276.trans k0
    · exact c4277.trans k0
    · exact c4278.trans k0
    · exact c4279.trans k0
    · exact c4280.trans k0
    · exact c4281.trans k0
    · exact c4282.trans k0
    · exact c4283.trans k0
    · exact c4284.trans k0
    · exact c4285.trans k0
    · exact c4286.trans k0
    · exact c4287.trans k0
    · exact c4288.trans k0
    · exact c4289.trans k0
    · exact c4290.trans k0
    · exact c4291.trans k0
    )
  rwa [c4292] at hr

theorem leafCircuit_f828 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 215)) (a (.virt 14)) (a (.virt 360)) (a (.virt 361)) := by
  have c4293 := (leafCircuit_copies134 a h).2.2.2.2.2.1
  have c4294 := (leafCircuit_copies134 a h).2.2.2.2.2.2.1
  have c4295 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.1
  have c4296 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.1
  have c4297 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.1
  have c4298 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4299 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4300 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4301 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4302 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4303 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4304 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4305 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4306 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_101_10 := arithEq_of_rows h (i := 10) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_10
  simp only [← c4296, ← c4297, ← c4298] at e_101_10
  have e_101_11 := arithEq_of_rows h (i := 11) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_11
  simp only [← c4299, ← c4300, ← c4301] at e_101_11
  have e_112_1 := arithEq_of_rows h (i := 1) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_1
  simp only [← c4293, k3, ← c4294, k3, ← c4295] at e_112_1
  have e_112_2 := arithEq_of_rows h (i := 2) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_2
  simp only [← c4302, ← c4303, k3, ← c4304] at e_112_2
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_101_10
    linear_combination c4305.trans k0 - hc
  · have hc := e_112_2
    simp only [e_112_1, e_101_11] at hc
    linear_combination c4306.trans k0 - hc

theorem leafCircuit_f829 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 215)) (a (.virt 292)) (a (.virt 362)) (a (.virt 363)) := by
  have c4307 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4308 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4309 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4310 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4311 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4312 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4313 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4314 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4315 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4316 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4317 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4318 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4319 := (leafCircuit_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4320 := (leafCircuit_copies135 a h).1
  have c4321 := (leafCircuit_copies135 a h).2.1
  have c4322 := (leafCircuit_copies135 a h).2.2.1
  have c4323 := (leafCircuit_copies135 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_101_12 := arithEq_of_rows h (i := 12) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_12
  simp only [← c4313, ← c4314, ← c4315] at e_101_12
  have e_101_13 := arithEq_of_rows h (i := 13) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_13
  simp only [← c4316, ← c4317, ← c4318] at e_101_13
  have e_112_3 := arithEq_of_rows h (i := 3) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_3
  simp only [← c4307, k3, ← c4308, k3, ← c4309] at e_112_3
  have e_112_4 := arithEq_of_rows h (i := 4) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_4
  simp only [← c4310, ← c4311, k3, ← c4312, k3] at e_112_4
  have e_112_5 := arithEq_of_rows h (i := 5) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_5
  simp only [← c4319, ← c4320, k3, ← c4321] at e_112_5
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_101_12
    simp only [e_112_4] at hc
    linear_combination c4322.trans k0 - hc
  · have hc := e_112_5
    simp only [e_112_3, e_101_13, e_112_4] at hc
    linear_combination c4323.trans k0 - hc

theorem leafCircuit_f830 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 215)) (a (.virt 294)) (a (.virt 364)) (a (.virt 365)) := by
  have c4324 := (leafCircuit_copies135 a h).2.2.2.2.1
  have c4325 := (leafCircuit_copies135 a h).2.2.2.2.2.1
  have c4326 := (leafCircuit_copies135 a h).2.2.2.2.2.2.1
  have c4327 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.1
  have c4328 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.1
  have c4329 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.1
  have c4330 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4331 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4332 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4333 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4334 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4335 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4336 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4337 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4338 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4339 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4340 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_101_14 := arithEq_of_rows h (i := 14) leafCircuit_row101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_14
  simp only [← c4330, ← c4331, ← c4332] at e_101_14
  have e_112_6 := arithEq_of_rows h (i := 6) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_6
  simp only [← c4324, k3, ← c4325, k3, ← c4326] at e_112_6
  have e_112_7 := arithEq_of_rows h (i := 7) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_7
  simp only [← c4327, ← c4328, k3, ← c4329, k5] at e_112_7
  have e_112_8 := arithEq_of_rows h (i := 8) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_8
  simp only [← c4336, ← c4337, k3, ← c4338] at e_112_8
  have e_114_0 := arithEq_of_rows h (i := 0) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_0
  simp only [← c4333, ← c4334, ← c4335] at e_114_0
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_101_14
    simp only [e_112_7] at hc
    linear_combination c4339.trans k0 - hc
  · have hc := e_112_8
    simp only [e_112_6, e_114_0, e_112_7] at hc
    linear_combination c4340.trans k0 - hc

theorem leafCircuit_f831 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 215)) (a (.virt 295)) (a (.virt 366)) (a (.virt 367)) := by
  have c4341 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4342 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4343 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4344 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4345 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4346 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4347 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4348 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4349 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4350 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4351 := (leafCircuit_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4352 := (leafCircuit_copies136 a h).1
  have c4353 := (leafCircuit_copies136 a h).2.1
  have c4354 := (leafCircuit_copies136 a h).2.2.1
  have c4355 := (leafCircuit_copies136 a h).2.2.2.1
  have c4356 := (leafCircuit_copies136 a h).2.2.2.2.1
  have c4357 := (leafCircuit_copies136 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_112_9 := arithEq_of_rows h (i := 9) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_9
  simp only [← c4341, k3, ← c4342, k3, ← c4343] at e_112_9
  have e_112_10 := arithEq_of_rows h (i := 10) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_10
  simp only [← c4344, ← c4345, k3, ← c4346, k6] at e_112_10
  have e_112_11 := arithEq_of_rows h (i := 11) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_11
  simp only [← c4353, ← c4354, k3, ← c4355] at e_112_11
  have e_114_1 := arithEq_of_rows h (i := 1) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_1
  simp only [← c4347, ← c4348, ← c4349] at e_114_1
  have e_114_2 := arithEq_of_rows h (i := 2) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_2
  simp only [← c4350, ← c4351, ← c4352] at e_114_2
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_114_1
    simp only [e_112_10] at hc
    linear_combination c4356.trans k0 - hc
  · have hc := e_112_11
    simp only [e_112_9, e_114_2, e_112_10] at hc
    linear_combination c4357.trans k0 - hc

theorem leafCircuit_f832 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 112 55) = bselect (a (.virt 360)) (a (.wire 106 15)) (a (.virt 111)) := by
  have c4358 := (leafCircuit_copies136 a h).2.2.2.2.2.2.1
  have c4359 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.1
  have c4360 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.1
  have c4361 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.1
  have c4362 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4363 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_12 := arithEq_of_rows h (i := 12) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_12
  simp only [← c4358, ← c4359, ← c4360] at e_112_12
  have e_112_13 := arithEq_of_rows h (i := 13) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_13
  simp only [← c4361, ← c4362, ← c4363] at e_112_13
  have hr := e_112_13
  simp only [e_112_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f833 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 3) = bselect (a (.virt 360)) (a (.wire 106 23)) (a (.virt 112)) := by
  have c4364 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4365 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4366 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4367 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4368 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4369 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_14 := arithEq_of_rows h (i := 14) leafCircuit_row112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_14
  simp only [← c4364, ← c4365, ← c4366] at e_112_14
  have e_115_0 := arithEq_of_rows h (i := 0) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_0
  simp only [← c4367, ← c4368, ← c4369] at e_115_0
  have hr := e_115_0
  simp only [e_112_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f834 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 11) = bselect (a (.virt 360)) (a (.wire 106 31)) (a (.virt 113)) := by
  have c4370 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4371 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4372 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4373 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4374 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4375 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_1 := arithEq_of_rows h (i := 1) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_1
  simp only [← c4370, ← c4371, ← c4372] at e_115_1
  have e_115_2 := arithEq_of_rows h (i := 2) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_2
  simp only [← c4373, ← c4374, ← c4375] at e_115_2
  have hr := e_115_2
  simp only [e_115_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f835 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 19) = bselect (a (.virt 360)) (a (.wire 106 39)) (a (.virt 114)) := by
  have c4376 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4377 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4378 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4379 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4380 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4381 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_3 := arithEq_of_rows h (i := 3) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_3
  simp only [← c4376, ← c4377, ← c4378] at e_115_3
  have e_115_4 := arithEq_of_rows h (i := 4) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_4
  simp only [← c4379, ← c4380, ← c4381] at e_115_4
  have hr := e_115_4
  simp only [e_115_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f836 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 27) = bselect (a (.virt 360)) (a (.virt 111)) (a (.virt 115)) := by
  have c4382 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4383 := (leafCircuit_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4384 := (leafCircuit_copies137 a h).1
  have c4385 := (leafCircuit_copies137 a h).2.1
  have c4386 := (leafCircuit_copies137 a h).2.2.1
  have c4387 := (leafCircuit_copies137 a h).2.2.2.1
  have e_115_5 := arithEq_of_rows h (i := 5) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_5
  simp only [← c4382, ← c4383, ← c4384] at e_115_5
  have e_115_6 := arithEq_of_rows h (i := 6) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_6
  simp only [← c4385, ← c4386, ← c4387] at e_115_6
  have hr := e_115_6
  simp only [e_115_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f837 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 35) = bselect (a (.virt 362)) (a (.wire 106 15)) (a (.wire 115 27)) := by
  have c4388 := (leafCircuit_copies137 a h).2.2.2.2.1
  have c4389 := (leafCircuit_copies137 a h).2.2.2.2.2.1
  have c4390 := (leafCircuit_copies137 a h).2.2.2.2.2.2.1
  have c4391 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.1
  have c4392 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.1
  have c4393 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.1
  have e_115_7 := arithEq_of_rows h (i := 7) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_7
  simp only [← c4388, ← c4389, ← c4390] at e_115_7
  have e_115_8 := arithEq_of_rows h (i := 8) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_8
  simp only [← c4391, ← c4392, ← c4393] at e_115_8
  have hr := e_115_8
  simp only [e_115_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f838 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 43) = bselect (a (.virt 360)) (a (.virt 112)) (a (.virt 116)) := by
  have c4394 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4395 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4396 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4397 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4398 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4399 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_9 := arithEq_of_rows h (i := 9) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_9
  simp only [← c4394, ← c4395, ← c4396] at e_115_9
  have e_115_10 := arithEq_of_rows h (i := 10) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_10
  simp only [← c4397, ← c4398, ← c4399] at e_115_10
  have hr := e_115_10
  simp only [e_115_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f839 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 51) = bselect (a (.virt 362)) (a (.wire 106 23)) (a (.wire 115 43)) := by
  have c4400 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4401 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4402 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4403 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4404 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4405 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_11 := arithEq_of_rows h (i := 11) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_11
  simp only [← c4400, ← c4401, ← c4402] at e_115_11
  have e_115_12 := arithEq_of_rows h (i := 12) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_12
  simp only [← c4403, ← c4404, ← c4405] at e_115_12
  have hr := e_115_12
  simp only [e_115_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f840 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 115 59) = bselect (a (.virt 360)) (a (.virt 113)) (a (.virt 117)) := by
  have c4406 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4407 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4408 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4409 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4410 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4411 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_13 := arithEq_of_rows h (i := 13) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_13
  simp only [← c4406, ← c4407, ← c4408] at e_115_13
  have e_115_14 := arithEq_of_rows h (i := 14) leafCircuit_row115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_14
  simp only [← c4409, ← c4410, ← c4411] at e_115_14
  have hr := e_115_14
  simp only [e_115_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f841 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 7) = bselect (a (.virt 362)) (a (.wire 106 31)) (a (.wire 115 59)) := by
  have c4412 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4413 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4414 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4415 := (leafCircuit_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4416 := (leafCircuit_copies138 a h).1
  have c4417 := (leafCircuit_copies138 a h).2.1
  have e_116_0 := arithEq_of_rows h (i := 0) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_0
  simp only [← c4412, ← c4413, ← c4414] at e_116_0
  have e_116_1 := arithEq_of_rows h (i := 1) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_1
  simp only [← c4415, ← c4416, ← c4417] at e_116_1
  have hr := e_116_1
  simp only [e_116_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f842 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 15) = bselect (a (.virt 360)) (a (.virt 114)) (a (.virt 118)) := by
  have c4418 := (leafCircuit_copies138 a h).2.2.1
  have c4419 := (leafCircuit_copies138 a h).2.2.2.1
  have c4420 := (leafCircuit_copies138 a h).2.2.2.2.1
  have c4421 := (leafCircuit_copies138 a h).2.2.2.2.2.1
  have c4422 := (leafCircuit_copies138 a h).2.2.2.2.2.2.1
  have c4423 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.1
  have e_116_2 := arithEq_of_rows h (i := 2) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_2
  simp only [← c4418, ← c4419, ← c4420] at e_116_2
  have e_116_3 := arithEq_of_rows h (i := 3) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_3
  simp only [← c4421, ← c4422, ← c4423] at e_116_3
  have hr := e_116_3
  simp only [e_116_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f843 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 23) = bselect (a (.virt 362)) (a (.wire 106 39)) (a (.wire 116 15)) := by
  have c4424 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.1
  have c4425 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.1
  have c4426 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4427 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4428 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4429 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_4 := arithEq_of_rows h (i := 4) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_4
  simp only [← c4424, ← c4425, ← c4426] at e_116_4
  have e_116_5 := arithEq_of_rows h (i := 5) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_5
  simp only [← c4427, ← c4428, ← c4429] at e_116_5
  have hr := e_116_5
  simp only [e_116_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f844 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) := by
  have c4430 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4432 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4435 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_4 := arithEq_of_rows h (i := 4) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c4430, ← c4431, ← c4432] at e_104_4
  have e_111_1 := arithEq_of_rows h (i := 1) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c4433, ← c4434, k3, ← c4435] at e_111_1
  have hr := e_111_1
  simp only [e_104_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f845 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 31) = bselect (a (.wire 111 7)) (a (.virt 115)) (a (.virt 119)) := by
  have c4436 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4437 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4438 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4439 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4440 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4441 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_6 := arithEq_of_rows h (i := 6) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_6
  simp only [← c4436, ← c4437, ← c4438] at e_116_6
  have e_116_7 := arithEq_of_rows h (i := 7) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_7
  simp only [← c4439, ← c4440, ← c4441] at e_116_7
  have hr := e_116_7
  simp only [e_116_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f846 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 39) = bselect (a (.virt 364)) (a (.wire 106 15)) (a (.wire 116 31)) := by
  have c4442 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4443 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4444 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4445 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4446 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4447 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_116_8 := arithEq_of_rows h (i := 8) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_8
  simp only [← c4442, ← c4443, ← c4444] at e_116_8
  have e_116_9 := arithEq_of_rows h (i := 9) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_9
  simp only [← c4445, ← c4446, ← c4447] at e_116_9
  have hr := e_116_9
  simp only [e_116_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f847 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) := by
  have c4430 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4432 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4435 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_4 := arithEq_of_rows h (i := 4) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c4430, ← c4431, ← c4432] at e_104_4
  have e_111_1 := arithEq_of_rows h (i := 1) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c4433, ← c4434, k3, ← c4435] at e_111_1
  have hr := e_111_1
  simp only [e_104_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f848 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 47) = bselect (a (.wire 111 7)) (a (.virt 116)) (a (.virt 120)) := by
  have c4448 := (leafCircuit_copies139 a h).1
  have c4449 := (leafCircuit_copies139 a h).2.1
  have c4450 := (leafCircuit_copies139 a h).2.2.1
  have c4451 := (leafCircuit_copies139 a h).2.2.2.1
  have c4452 := (leafCircuit_copies139 a h).2.2.2.2.1
  have c4453 := (leafCircuit_copies139 a h).2.2.2.2.2.1
  have e_116_10 := arithEq_of_rows h (i := 10) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_10
  simp only [← c4448, ← c4449, ← c4450] at e_116_10
  have e_116_11 := arithEq_of_rows h (i := 11) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_11
  simp only [← c4451, ← c4452, ← c4453] at e_116_11
  have hr := e_116_11
  simp only [e_116_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f849 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 116 55) = bselect (a (.virt 364)) (a (.wire 106 23)) (a (.wire 116 47)) := by
  have c4454 := (leafCircuit_copies139 a h).2.2.2.2.2.2.1
  have c4455 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.1
  have c4456 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.1
  have c4457 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.1
  have c4458 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4459 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_12 := arithEq_of_rows h (i := 12) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_12
  simp only [← c4454, ← c4455, ← c4456] at e_116_12
  have e_116_13 := arithEq_of_rows h (i := 13) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_13
  simp only [← c4457, ← c4458, ← c4459] at e_116_13
  have hr := e_116_13
  simp only [e_116_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f850 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) := by
  have c4430 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4432 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4435 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_4 := arithEq_of_rows h (i := 4) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c4430, ← c4431, ← c4432] at e_104_4
  have e_111_1 := arithEq_of_rows h (i := 1) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c4433, ← c4434, k3, ← c4435] at e_111_1
  have hr := e_111_1
  simp only [e_104_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f851 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 3) = bselect (a (.wire 111 7)) (a (.virt 117)) (a (.virt 121)) := by
  have c4460 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4461 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4462 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4463 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4464 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4465 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_14 := arithEq_of_rows h (i := 14) leafCircuit_row116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_14
  simp only [← c4460, ← c4461, ← c4462] at e_116_14
  have e_117_0 := arithEq_of_rows h (i := 0) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_0
  simp only [← c4463, ← c4464, ← c4465] at e_117_0
  have hr := e_117_0
  simp only [e_116_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f852 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 11) = bselect (a (.virt 364)) (a (.wire 106 31)) (a (.wire 117 3)) := by
  have c4466 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4467 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4468 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4469 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4470 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4471 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_1 := arithEq_of_rows h (i := 1) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_1
  simp only [← c4466, ← c4467, ← c4468] at e_117_1
  have e_117_2 := arithEq_of_rows h (i := 2) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_2
  simp only [← c4469, ← c4470, ← c4471] at e_117_2
  have hr := e_117_2
  simp only [e_117_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f853 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) := by
  have c4430 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4432 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4435 := (leafCircuit_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_4 := arithEq_of_rows h (i := 4) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c4430, ← c4431, ← c4432] at e_104_4
  have e_111_1 := arithEq_of_rows h (i := 1) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c4433, ← c4434, k3, ← c4435] at e_111_1
  have hr := e_111_1
  simp only [e_104_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f854 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 19) = bselect (a (.wire 111 7)) (a (.virt 118)) (a (.virt 122)) := by
  have c4472 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4473 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4474 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4475 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4476 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4477 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_3 := arithEq_of_rows h (i := 3) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_3
  simp only [← c4472, ← c4473, ← c4474] at e_117_3
  have e_117_4 := arithEq_of_rows h (i := 4) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_4
  simp only [← c4475, ← c4476, ← c4477] at e_117_4
  have hr := e_117_4
  simp only [e_117_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f855 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 27) = bselect (a (.virt 364)) (a (.wire 106 39)) (a (.wire 117 19)) := by
  have c4478 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4479 := (leafCircuit_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4480 := (leafCircuit_copies140 a h).1
  have c4481 := (leafCircuit_copies140 a h).2.1
  have c4482 := (leafCircuit_copies140 a h).2.2.1
  have c4483 := (leafCircuit_copies140 a h).2.2.2.1
  have e_117_5 := arithEq_of_rows h (i := 5) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_5
  simp only [← c4478, ← c4479, ← c4480] at e_117_5
  have e_117_6 := arithEq_of_rows h (i := 6) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_6
  simp only [← c4481, ← c4482, ← c4483] at e_117_6
  have hr := e_117_6
  simp only [e_117_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f856 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 35) = bselect (a (.virt 366)) (a (.wire 106 15)) (a (.virt 119)) := by
  have c4484 := (leafCircuit_copies140 a h).2.2.2.2.1
  have c4485 := (leafCircuit_copies140 a h).2.2.2.2.2.1
  have c4486 := (leafCircuit_copies140 a h).2.2.2.2.2.2.1
  have c4487 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.1
  have c4488 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.1
  have c4489 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.1
  have e_117_7 := arithEq_of_rows h (i := 7) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_7
  simp only [← c4484, ← c4485, ← c4486] at e_117_7
  have e_117_8 := arithEq_of_rows h (i := 8) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_8
  simp only [← c4487, ← c4488, ← c4489] at e_117_8
  have hr := e_117_8
  simp only [e_117_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f857 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 43) = bselect (a (.virt 366)) (a (.wire 106 23)) (a (.virt 120)) := by
  have c4490 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4491 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4492 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4493 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4494 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4495 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_9 := arithEq_of_rows h (i := 9) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_9
  simp only [← c4490, ← c4491, ← c4492] at e_117_9
  have e_117_10 := arithEq_of_rows h (i := 10) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_10
  simp only [← c4493, ← c4494, ← c4495] at e_117_10
  have hr := e_117_10
  simp only [e_117_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f858 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 51) = bselect (a (.virt 366)) (a (.wire 106 31)) (a (.virt 121)) := by
  have c4496 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4497 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4498 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4499 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4500 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4501 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_11 := arithEq_of_rows h (i := 11) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_11
  simp only [← c4496, ← c4497, ← c4498] at e_117_11
  have e_117_12 := arithEq_of_rows h (i := 12) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_12
  simp only [← c4499, ← c4500, ← c4501] at e_117_12
  have hr := e_117_12
  simp only [e_117_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f859 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 117 59) = bselect (a (.virt 366)) (a (.wire 106 39)) (a (.virt 122)) := by
  have c4502 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4503 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4504 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4505 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4506 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4507 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_13 := arithEq_of_rows h (i := 13) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_13
  simp only [← c4502, ← c4503, ← c4504] at e_117_13
  have e_117_14 := arithEq_of_rows h (i := 14) leafCircuit_row117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_14
  simp only [← c4505, ← c4506, ← c4507] at e_117_14
  have hr := e_117_14
  simp only [e_117_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f860 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 120 12) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 0 ∧ a (.wire 120 13) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 1 ∧ a (.wire 120 14) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 2 ∧ a (.wire 120 15) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 3) := by
  have c4508 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4509 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4510 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4511 := (leafCircuit_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4512 := (leafCircuit_copies141 a h).1
  have c4513 := (leafCircuit_copies141 a h).2.1
  have c4514 := (leafCircuit_copies141 a h).2.2.1
  have c4515 := (leafCircuit_copies141 a h).2.2.2.1
  have c4516 := (leafCircuit_copies141 a h).2.2.2.2.1
  have c4517 := (leafCircuit_copies141 a h).2.2.2.2.2.1
  have c4518 := (leafCircuit_copies141 a h).2.2.2.2.2.2.1
  have c4519 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.1
  have c4520 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.1
  have c4521 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.1
  have c4522 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4523 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4524 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4525 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4526 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4527 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4528 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4529 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4530 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4531 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4532 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4533 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4534 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4535 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4536 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4537 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4538 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4539 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4540 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4541 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4542 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4543 := (leafCircuit_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4544 := (leafCircuit_copies142 a h).1
  have c4545 := (leafCircuit_copies142 a h).2.1
  have c4546 := (leafCircuit_copies142 a h).2.2.1
  have c4547 := (leafCircuit_copies142 a h).2.2.2.1
  have c4548 := (leafCircuit_copies142 a h).2.2.2.2.1
  have c4549 := (leafCircuit_copies142 a h).2.2.2.2.2.1
  have c4550 := (leafCircuit_copies142 a h).2.2.2.2.2.2.1
  have c4551 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.1
  have c4552 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.1
  have c4553 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.1
  have c4554 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4555 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4556 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4557 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4558 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4559 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4560 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4561 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4562 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4563 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4564 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4565 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4566 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4567 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4568 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4569 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4570 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_111_2 := arithEq_of_rows h (i := 2) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_2
  simp only [← c4520, ← c4521, k3, ← c4522] at e_111_2
  have e_111_3 := arithEq_of_rows h (i := 3) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_3
  simp only [← c4523, ← c4524, k3, ← c4525] at e_111_3
  have e_111_4 := arithEq_of_rows h (i := 4) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_4
  simp only [← c4526, ← c4527, k3, ← c4528] at e_111_4
  have e_111_5 := arithEq_of_rows h (i := 5) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_5
  simp only [← c4529, ← c4530, k3, ← c4531] at e_111_5
  have e_111_6 := arithEq_of_rows h (i := 6) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_6
  simp only [← c4532, ← c4533, k3, ← c4534] at e_111_6
  have e_111_7 := arithEq_of_rows h (i := 7) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_7
  simp only [← c4535, ← c4536, k3, ← c4537] at e_111_7
  have e_111_8 := arithEq_of_rows h (i := 8) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_8
  simp only [← c4538, ← c4539, k3, ← c4540] at e_111_8
  have e_111_9 := arithEq_of_rows h (i := 9) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_9
  simp only [← c4541, ← c4542, k3, ← c4543] at e_111_9
  have e_111_10 := arithEq_of_rows h (i := 10) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_10
  simp only [← c4556, ← c4557, k3, ← c4558, k3] at e_111_10
  have hin0 : poseidon2In a 118 = addBlock (fun _ => 0) [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23)] :=
    poseidon2In_first c4508.symm c4509.symm c4510.symm c4511.symm c4512.symm c4513.symm c4514.symm c4515.symm (c4516.symm.trans k0) (c4517.symm.trans k0) (c4518.symm.trans k0) (c4519.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 118) rfl rfl hin0
  have hin1 : poseidon2In a 119 = addBlock (poseidon2Out a 118) [a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] :=
    poseidon2In_chain (by linear_combination -c4544 + e_111_2) (by linear_combination -c4545 + e_111_3) (by linear_combination -c4546 + e_111_4) (by linear_combination -c4547 + e_111_5) (by linear_combination -c4548 + e_111_6) (by linear_combination -c4549 + e_111_7) (by linear_combination -c4550 + e_111_8) (by linear_combination -c4551 + e_111_9) c4552.symm c4553.symm c4554.symm c4555.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 119) rfl rfl hin1
  have hin2 : poseidon2In a 120 = addBlock (poseidon2Out a 119) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c4559 + e_111_10) (c4560.symm.trans (add_zero _).symm) (c4561.symm.trans (add_zero _).symm) (c4562.symm.trans (add_zero _).symm) (c4563.symm.trans (add_zero _).symm) (c4564.symm.trans (add_zero _).symm) (c4565.symm.trans (add_zero _).symm) (c4566.symm.trans (add_zero _).symm) c4567.symm c4568.symm c4569.symm c4570.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 120) rfl rfl hin2
  have hpad : pad10 [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] =
      [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23)] ++ ([a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f861 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 7) = bselect (a (.wire 111 3)) (a (.wire 120 12)) (a (.wire 106 15)) := by
  have c4571 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4572 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4573 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4574 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4575 := (leafCircuit_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4576 := (leafCircuit_copies143 a h).1
  have e_121_0 := arithEq_of_rows h (i := 0) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_0
  simp only [← c4571, ← c4572, ← c4573] at e_121_0
  have e_121_1 := arithEq_of_rows h (i := 1) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_1
  simp only [← c4574, ← c4575, ← c4576] at e_121_1
  have hr := e_121_1
  simp only [e_121_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f862 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 15) = bselect (a (.wire 111 3)) (a (.wire 120 13)) (a (.wire 106 23)) := by
  have c4577 := (leafCircuit_copies143 a h).2.1
  have c4578 := (leafCircuit_copies143 a h).2.2.1
  have c4579 := (leafCircuit_copies143 a h).2.2.2.1
  have c4580 := (leafCircuit_copies143 a h).2.2.2.2.1
  have c4581 := (leafCircuit_copies143 a h).2.2.2.2.2.1
  have c4582 := (leafCircuit_copies143 a h).2.2.2.2.2.2.1
  have e_121_2 := arithEq_of_rows h (i := 2) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_2
  simp only [← c4577, ← c4578, ← c4579] at e_121_2
  have e_121_3 := arithEq_of_rows h (i := 3) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_3
  simp only [← c4580, ← c4581, ← c4582] at e_121_3
  have hr := e_121_3
  simp only [e_121_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f863 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 23) = bselect (a (.wire 111 3)) (a (.wire 120 14)) (a (.wire 106 31)) := by
  have c4583 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.1
  have c4584 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.1
  have c4585 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.1
  have c4586 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4587 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4588 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_4 := arithEq_of_rows h (i := 4) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_4
  simp only [← c4583, ← c4584, ← c4585] at e_121_4
  have e_121_5 := arithEq_of_rows h (i := 5) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_5
  simp only [← c4586, ← c4587, ← c4588] at e_121_5
  have hr := e_121_5
  simp only [e_121_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f864 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 31) = bselect (a (.wire 111 3)) (a (.wire 120 15)) (a (.wire 106 39)) := by
  have c4589 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4590 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4591 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4592 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4593 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4594 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_6 := arithEq_of_rows h (i := 6) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_6
  simp only [← c4589, ← c4590, ← c4591] at e_121_6
  have e_121_7 := arithEq_of_rows h (i := 7) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_7
  simp only [← c4592, ← c4593, ← c4594] at e_121_7
  have hr := e_121_7
  simp only [e_121_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f865 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 122 1), a (.wire 122 2), a (.wire 122 3), a (.wire 122 4), a (.wire 122 5)] := by
  have c4595 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4596 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4597 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4598 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4599 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4600 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4601 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4602 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4603 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4604 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4605 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4606 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4607 := (leafCircuit_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4608 := (leafCircuit_copies144 a h).1
  have c4609 := (leafCircuit_copies144 a h).2.1
  have c4610 := (leafCircuit_copies144 a h).2.2.1
  have c4611 := (leafCircuit_copies144 a h).2.2.2.1
  have c4612 := (leafCircuit_copies144 a h).2.2.2.2.1
  have c4613 := (leafCircuit_copies144 a h).2.2.2.2.2.1
  have c4614 := (leafCircuit_copies144 a h).2.2.2.2.2.2.1
  have c4615 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.1
  have c4616 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.1
  have c4617 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.1
  have c4618 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4619 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4620 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4621 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4622 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4623 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4624 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4625 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4626 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4627 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4628 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4629 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4630 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4631 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4632 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4633 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4634 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4635 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4636 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4637 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4638 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4639 := (leafCircuit_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4640 := (leafCircuit_copies145 a h).1
  have c4641 := (leafCircuit_copies145 a h).2.1
  have c4642 := (leafCircuit_copies145 a h).2.2.1
  have c4643 := (leafCircuit_copies145 a h).2.2.2.1
  have c4644 := (leafCircuit_copies145 a h).2.2.2.2.1
  have c4645 := (leafCircuit_copies145 a h).2.2.2.2.2.1
  have c4646 := (leafCircuit_copies145 a h).2.2.2.2.2.2.1
  have c4647 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.1
  have c4648 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.1
  have c4649 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 122) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c4595.trans k0
    · exact c4596.trans k0
    · exact c4597.trans k0
    · exact c4598.trans k0
    · exact c4599.trans k0
    · exact c4600.trans k0
    · exact c4601.trans k0
    · exact c4602.trans k0
    · exact c4603.trans k0
    · exact c4604.trans k0
    · exact c4605.trans k0
    · exact c4606.trans k0
    · exact c4607.trans k0
    · exact c4608.trans k0
    · exact c4609.trans k0
    · exact c4610.trans k0
    · exact c4611.trans k0
    · exact c4612.trans k0
    · exact c4613.trans k0
    · exact c4614.trans k0
    · exact c4615.trans k0
    · exact c4616.trans k0
    · exact c4617.trans k0
    · exact c4618.trans k0
    · exact c4619.trans k0
    · exact c4620.trans k0
    · exact c4621.trans k0
    · exact c4622.trans k0
    · exact c4623.trans k0
    · exact c4624.trans k0
    · exact c4625.trans k0
    · exact c4626.trans k0
    · exact c4627.trans k0
    · exact c4628.trans k0
    · exact c4629.trans k0
    · exact c4630.trans k0
    · exact c4631.trans k0
    · exact c4632.trans k0
    · exact c4633.trans k0
    · exact c4634.trans k0
    · exact c4635.trans k0
    · exact c4636.trans k0
    · exact c4637.trans k0
    · exact c4638.trans k0
    · exact c4639.trans k0
    · exact c4640.trans k0
    · exact c4641.trans k0
    · exact c4642.trans k0
    · exact c4643.trans k0
    · exact c4644.trans k0
    · exact c4645.trans k0
    · exact c4646.trans k0
    · exact c4647.trans k0
    · exact c4648.trans k0
    )
  rwa [c4649] at hr

theorem leafCircuit_f866 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f867 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 5) = band (a (.virt 292)) (a (.wire 122 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f868 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 5) = band (a (.wire 122 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f869 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 5) = bor (a (.virt 14)) (a (.wire 122 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f870 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 122 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f871 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f872 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 5) = a (.virt 14) + a (.wire 122 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f873 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 5) = a (.wire 122 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f874 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 35) = bnot (a (.wire 122 5)) := by
  have c4650 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4651 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4652 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_8 := arithEq_of_rows h (i := 8) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_8
  simp only [← c4650, k3, ← c4651, k3, ← c4652] at e_121_8
  have hr := e_121_8
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f875 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 35) = band (a (.virt 292)) (a (.wire 121 35)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f876 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f877 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 122 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f878 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 121 35)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f879 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 5) = bor (a (.wire 122 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f880 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 4) = a (.virt 292) * a (.wire 122 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f881 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 15) = a (.virt 294) * a (.wire 122 4) := by
  have c4653 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4654 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4655 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_114_3 := arithEq_of_rows h (i := 3) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_3
  simp only [← c4653, k5, ← c4654, ← c4655, k5] at e_114_3
  have hr := e_114_3
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f882 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 47) = a (.virt 292) + a (.wire 122 4) := by
  have c4656 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4657 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4658 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_111_11 := arithEq_of_rows h (i := 11) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_11
  simp only [← c4656, k3, ← c4657, k3, ← c4658] at e_111_11
  have hr := e_111_11
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f883 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 39) = a (.wire 111 47) - a (.wire 114 15) := by
  have c4659 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4660 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4661 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_9 := arithEq_of_rows h (i := 9) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_9
  simp only [← c4659, ← c4660, k3, ← c4661] at e_121_9
  have hr := e_121_9
  linear_combination hr

theorem leafCircuit_f884 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 43) = bnot (a (.wire 121 39)) := by
  have c4662 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4663 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4664 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_10 := arithEq_of_rows h (i := 10) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_10
  simp only [← c4662, k3, ← c4663, k3, ← c4664] at e_121_10
  have hr := e_121_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f885 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 19) = band (a (.wire 121 35)) (a (.wire 121 43)) := by
  have c4665 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4666 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4667 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_4 := arithEq_of_rows h (i := 4) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_4
  simp only [← c4665, ← c4666, ← c4667] at e_114_4
  have hr := e_114_4
  simp only [band]
  linear_combination hr

theorem leafCircuit_f886 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f887 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 3) = band (a (.virt 292)) (a (.wire 122 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f888 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 23) = band (a (.wire 122 3)) (a (.wire 114 19)) := by
  have c4668 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4669 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4670 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_5 := arithEq_of_rows h (i := 5) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_5
  simp only [← c4668, ← c4669, ← c4670] at e_114_5
  have hr := e_114_5
  simp only [band]
  linear_combination hr

theorem leafCircuit_f889 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 51) = bor (a (.wire 122 5)) (a (.wire 114 23)) := by
  have c4671 := (leafCircuit_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4672 := (leafCircuit_copies146 a h).1
  have c4673 := (leafCircuit_copies146 a h).2.1
  have c4674 := (leafCircuit_copies146 a h).2.2.1
  have c4675 := (leafCircuit_copies146 a h).2.2.2.1
  have c4676 := (leafCircuit_copies146 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_5 := arithEq_of_rows h (i := 5) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_5
  simp only [← c4671, ← c4672, ← c4673] at e_104_5
  have e_111_12 := arithEq_of_rows h (i := 12) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_12
  simp only [← c4674, ← c4675, k3, ← c4676] at e_111_12
  have hr := e_111_12
  simp only [e_104_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f890 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 122 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f891 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f892 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 3) = a (.virt 14) + a (.wire 122 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f893 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 3) = a (.wire 122 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f894 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 47) = bnot (a (.wire 122 3)) := by
  have c4677 := (leafCircuit_copies146 a h).2.2.2.2.2.1
  have c4678 := (leafCircuit_copies146 a h).2.2.2.2.2.2.1
  have c4679 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_11 := arithEq_of_rows h (i := 11) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_11
  simp only [← c4677, k3, ← c4678, k3, ← c4679] at e_121_11
  have hr := e_121_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f895 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 27) = band (a (.wire 114 19)) (a (.wire 121 47)) := by
  have c4680 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.1
  have c4681 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.1
  have c4682 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_114_6 := arithEq_of_rows h (i := 6) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_6
  simp only [← c4680, ← c4681, ← c4682] at e_114_6
  have hr := e_114_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f896 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f897 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 2) = band (a (.virt 292)) (a (.wire 122 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f898 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 31) = band (a (.wire 122 2)) (a (.wire 114 27)) := by
  have c4683 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4684 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4685 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_7 := arithEq_of_rows h (i := 7) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_7
  simp only [← c4683, ← c4684, ← c4685] at e_114_7
  have hr := e_114_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f899 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 55) = bor (a (.wire 111 51)) (a (.wire 114 31)) := by
  have c4686 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4687 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4688 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4689 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4690 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4691 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_6 := arithEq_of_rows h (i := 6) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_6
  simp only [← c4686, ← c4687, ← c4688] at e_104_6
  have e_111_13 := arithEq_of_rows h (i := 13) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_13
  simp only [← c4689, ← c4690, k3, ← c4691] at e_111_13
  have hr := e_111_13
  simp only [e_104_6] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f900 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 122 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f901 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f902 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 2) = a (.virt 14) + a (.wire 122 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f903 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 2) = a (.wire 122 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f904 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 51) = bnot (a (.wire 122 2)) := by
  have c4692 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4693 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4694 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_12 := arithEq_of_rows h (i := 12) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_12
  simp only [← c4692, k3, ← c4693, k3, ← c4694] at e_121_12
  have hr := e_121_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f905 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 35) = band (a (.wire 114 27)) (a (.wire 121 51)) := by
  have c4695 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4696 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4697 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_8 := arithEq_of_rows h (i := 8) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_8
  simp only [← c4695, ← c4696, ← c4697] at e_114_8
  have hr := e_114_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f906 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f907 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 122 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f908 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 114 35)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f909 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 55) = bor (a (.wire 111 55)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f910 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 122 1) = a (.virt 292) * a (.wire 122 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f911 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 39) = a (.virt 294) * a (.wire 122 1) := by
  have c4698 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4699 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4700 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_114_9 := arithEq_of_rows h (i := 9) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_9
  simp only [← c4698, k5, ← c4699, ← c4700, k5] at e_114_9
  have hr := e_114_9
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f912 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 111 59) = a (.virt 292) + a (.wire 122 1) := by
  have c4701 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4702 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4703 := (leafCircuit_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_111_14 := arithEq_of_rows h (i := 14) leafCircuit_row111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_14
  simp only [← c4701, k3, ← c4702, k3, ← c4703] at e_111_14
  have hr := e_111_14
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f913 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 55) = a (.wire 111 59) - a (.wire 114 39) := by
  have c4704 := (leafCircuit_copies147 a h).1
  have c4705 := (leafCircuit_copies147 a h).2.1
  have c4706 := (leafCircuit_copies147 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_13 := arithEq_of_rows h (i := 13) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_13
  simp only [← c4704, ← c4705, k3, ← c4706] at e_121_13
  have hr := e_121_13
  linear_combination hr

theorem leafCircuit_f914 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 121 59) = bnot (a (.wire 121 55)) := by
  have c4707 := (leafCircuit_copies147 a h).2.2.2.1
  have c4708 := (leafCircuit_copies147 a h).2.2.2.2.1
  have c4709 := (leafCircuit_copies147 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_121_14 := arithEq_of_rows h (i := 14) leafCircuit_row121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_14
  simp only [← c4707, k3, ← c4708, k3, ← c4709] at e_121_14
  have hr := e_121_14
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f915 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 114 43) = band (a (.wire 114 35)) (a (.wire 121 59)) := by
  have c4710 := (leafCircuit_copies147 a h).2.2.2.2.2.2.1
  have c4711 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.1
  have c4712 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.1
  have e_114_10 := arithEq_of_rows h (i := 10) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_10
  simp only [← c4710, ← c4711, ← c4712] at e_114_10
  have hr := e_114_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f916 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 216)) 2 := by
  have c4713 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.1
  have c4714 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4715 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4716 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4717 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4718 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4719 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4720 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4721 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4722 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4723 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4724 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4725 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4726 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4727 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4728 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4729 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4730 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4731 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4732 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4733 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4734 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4735 := (leafCircuit_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4736 := (leafCircuit_copies148 a h).1
  have c4737 := (leafCircuit_copies148 a h).2.1
  have c4738 := (leafCircuit_copies148 a h).2.2.1
  have c4739 := (leafCircuit_copies148 a h).2.2.2.1
  have c4740 := (leafCircuit_copies148 a h).2.2.2.2.1
  have c4741 := (leafCircuit_copies148 a h).2.2.2.2.2.1
  have c4742 := (leafCircuit_copies148 a h).2.2.2.2.2.2.1
  have c4743 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.1
  have c4744 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.1
  have c4745 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.1
  have c4746 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4747 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4748 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4749 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4750 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4751 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4752 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4753 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4754 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4755 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4756 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4757 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4758 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4759 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4760 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4761 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4762 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4763 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4764 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4765 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4766 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4767 := (leafCircuit_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4768 := (leafCircuit_copies149 a h).1
  have c4769 := (leafCircuit_copies149 a h).2.1
  have c4770 := (leafCircuit_copies149 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 123) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c4713.trans k0
    · exact c4714.trans k0
    · exact c4715.trans k0
    · exact c4716.trans k0
    · exact c4717.trans k0
    · exact c4718.trans k0
    · exact c4719.trans k0
    · exact c4720.trans k0
    · exact c4721.trans k0
    · exact c4722.trans k0
    · exact c4723.trans k0
    · exact c4724.trans k0
    · exact c4725.trans k0
    · exact c4726.trans k0
    · exact c4727.trans k0
    · exact c4728.trans k0
    · exact c4729.trans k0
    · exact c4730.trans k0
    · exact c4731.trans k0
    · exact c4732.trans k0
    · exact c4733.trans k0
    · exact c4734.trans k0
    · exact c4735.trans k0
    · exact c4736.trans k0
    · exact c4737.trans k0
    · exact c4738.trans k0
    · exact c4739.trans k0
    · exact c4740.trans k0
    · exact c4741.trans k0
    · exact c4742.trans k0
    · exact c4743.trans k0
    · exact c4744.trans k0
    · exact c4745.trans k0
    · exact c4746.trans k0
    · exact c4747.trans k0
    · exact c4748.trans k0
    · exact c4749.trans k0
    · exact c4750.trans k0
    · exact c4751.trans k0
    · exact c4752.trans k0
    · exact c4753.trans k0
    · exact c4754.trans k0
    · exact c4755.trans k0
    · exact c4756.trans k0
    · exact c4757.trans k0
    · exact c4758.trans k0
    · exact c4759.trans k0
    · exact c4760.trans k0
    · exact c4761.trans k0
    · exact c4762.trans k0
    · exact c4763.trans k0
    · exact c4764.trans k0
    · exact c4765.trans k0
    · exact c4766.trans k0
    · exact c4767.trans k0
    · exact c4768.trans k0
    · exact c4769.trans k0
    )
  rwa [c4770] at hr

theorem leafCircuit_f917 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 216)) (a (.virt 14)) (a (.virt 368)) (a (.virt 369)) := by
  have c4771 := (leafCircuit_copies149 a h).2.2.2.1
  have c4772 := (leafCircuit_copies149 a h).2.2.2.2.1
  have c4773 := (leafCircuit_copies149 a h).2.2.2.2.2.1
  have c4774 := (leafCircuit_copies149 a h).2.2.2.2.2.2.1
  have c4775 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.1
  have c4776 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.1
  have c4777 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.1
  have c4778 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4779 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4780 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4781 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4782 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4783 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4784 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_114_11 := arithEq_of_rows h (i := 11) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_11
  simp only [← c4774, ← c4775, ← c4776] at e_114_11
  have e_114_12 := arithEq_of_rows h (i := 12) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_12
  simp only [← c4777, ← c4778, ← c4779] at e_114_12
  have e_124_0 := arithEq_of_rows h (i := 0) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_0
  simp only [← c4771, k3, ← c4772, k3, ← c4773] at e_124_0
  have e_124_1 := arithEq_of_rows h (i := 1) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_1
  simp only [← c4780, ← c4781, k3, ← c4782] at e_124_1
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_114_11
    linear_combination c4783.trans k0 - hc
  · have hc := e_124_1
    simp only [e_124_0, e_114_12] at hc
    linear_combination c4784.trans k0 - hc

theorem leafCircuit_f918 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 216)) (a (.virt 292)) (a (.virt 370)) (a (.virt 371)) := by
  have c4785 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4786 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4787 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4788 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4789 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4790 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4791 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4792 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4793 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4794 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4795 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4796 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4797 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4798 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4799 := (leafCircuit_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4800 := (leafCircuit_copies150 a h).1
  have c4801 := (leafCircuit_copies150 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_114_13 := arithEq_of_rows h (i := 13) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_13
  simp only [← c4791, ← c4792, ← c4793] at e_114_13
  have e_114_14 := arithEq_of_rows h (i := 14) leafCircuit_row114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_14
  simp only [← c4794, ← c4795, ← c4796] at e_114_14
  have e_124_2 := arithEq_of_rows h (i := 2) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_2
  simp only [← c4785, k3, ← c4786, k3, ← c4787] at e_124_2
  have e_124_3 := arithEq_of_rows h (i := 3) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_3
  simp only [← c4788, ← c4789, k3, ← c4790, k3] at e_124_3
  have e_124_4 := arithEq_of_rows h (i := 4) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_4
  simp only [← c4797, ← c4798, k3, ← c4799] at e_124_4
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_114_13
    simp only [e_124_3] at hc
    linear_combination c4800.trans k0 - hc
  · have hc := e_124_4
    simp only [e_124_2, e_114_14, e_124_3] at hc
    linear_combination c4801.trans k0 - hc

theorem leafCircuit_f919 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 216)) (a (.virt 294)) (a (.virt 372)) (a (.virt 373)) := by
  have c4802 := (leafCircuit_copies150 a h).2.2.1
  have c4803 := (leafCircuit_copies150 a h).2.2.2.1
  have c4804 := (leafCircuit_copies150 a h).2.2.2.2.1
  have c4805 := (leafCircuit_copies150 a h).2.2.2.2.2.1
  have c4806 := (leafCircuit_copies150 a h).2.2.2.2.2.2.1
  have c4807 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.1
  have c4808 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.1
  have c4809 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.1
  have c4810 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4811 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4812 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4813 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4814 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4815 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4816 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4817 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4818 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_124_5 := arithEq_of_rows h (i := 5) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_5
  simp only [← c4802, k3, ← c4803, k3, ← c4804] at e_124_5
  have e_124_6 := arithEq_of_rows h (i := 6) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_6
  simp only [← c4805, ← c4806, k3, ← c4807, k5] at e_124_6
  have e_124_7 := arithEq_of_rows h (i := 7) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_7
  simp only [← c4814, ← c4815, k3, ← c4816] at e_124_7
  have e_125_0 := arithEq_of_rows h (i := 0) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_0
  simp only [← c4808, ← c4809, ← c4810] at e_125_0
  have e_125_1 := arithEq_of_rows h (i := 1) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_1
  simp only [← c4811, ← c4812, ← c4813] at e_125_1
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_125_0
    simp only [e_124_6] at hc
    linear_combination c4817.trans k0 - hc
  · have hc := e_124_7
    simp only [e_124_5, e_125_1, e_124_6] at hc
    linear_combination c4818.trans k0 - hc

theorem leafCircuit_f920 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 216)) (a (.virt 295)) (a (.virt 374)) (a (.virt 375)) := by
  have c4819 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4820 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4821 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4822 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4823 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4824 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4825 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4826 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4827 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4828 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4829 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4830 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4831 := (leafCircuit_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4832 := (leafCircuit_copies151 a h).1
  have c4833 := (leafCircuit_copies151 a h).2.1
  have c4834 := (leafCircuit_copies151 a h).2.2.1
  have c4835 := (leafCircuit_copies151 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_124_8 := arithEq_of_rows h (i := 8) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_8
  simp only [← c4819, k3, ← c4820, k3, ← c4821] at e_124_8
  have e_124_9 := arithEq_of_rows h (i := 9) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_9
  simp only [← c4822, ← c4823, k3, ← c4824, k6] at e_124_9
  have e_124_10 := arithEq_of_rows h (i := 10) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_10
  simp only [← c4831, ← c4832, k3, ← c4833] at e_124_10
  have e_125_2 := arithEq_of_rows h (i := 2) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_2
  simp only [← c4825, ← c4826, ← c4827] at e_125_2
  have e_125_3 := arithEq_of_rows h (i := 3) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_3
  simp only [← c4828, ← c4829, ← c4830] at e_125_3
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_125_2
    simp only [e_124_9] at hc
    linear_combination c4834.trans k0 - hc
  · have hc := e_124_10
    simp only [e_124_8, e_125_3, e_124_9] at hc
    linear_combination c4835.trans k0 - hc

theorem leafCircuit_f921 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 124 51) = bselect (a (.virt 368)) (a (.wire 121 7)) (a (.virt 123)) := by
  have c4836 := (leafCircuit_copies151 a h).2.2.2.2.1
  have c4837 := (leafCircuit_copies151 a h).2.2.2.2.2.1
  have c4838 := (leafCircuit_copies151 a h).2.2.2.2.2.2.1
  have c4839 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.1
  have c4840 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.1
  have c4841 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.1
  have e_124_11 := arithEq_of_rows h (i := 11) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_11
  simp only [← c4836, ← c4837, ← c4838] at e_124_11
  have e_124_12 := arithEq_of_rows h (i := 12) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_12
  simp only [← c4839, ← c4840, ← c4841] at e_124_12
  have hr := e_124_12
  simp only [e_124_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f922 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 124 59) = bselect (a (.virt 368)) (a (.wire 121 15)) (a (.virt 124)) := by
  have c4842 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4843 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4844 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4845 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4846 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4847 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_13 := arithEq_of_rows h (i := 13) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_13
  simp only [← c4842, ← c4843, ← c4844] at e_124_13
  have e_124_14 := arithEq_of_rows h (i := 14) leafCircuit_row124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_14
  simp only [← c4845, ← c4846, ← c4847] at e_124_14
  have hr := e_124_14
  simp only [e_124_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f923 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 7) = bselect (a (.virt 368)) (a (.wire 121 23)) (a (.virt 125)) := by
  have c4848 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4849 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4850 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4851 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4852 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4853 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_0 := arithEq_of_rows h (i := 0) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_0
  simp only [← c4848, ← c4849, ← c4850] at e_126_0
  have e_126_1 := arithEq_of_rows h (i := 1) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_1
  simp only [← c4851, ← c4852, ← c4853] at e_126_1
  have hr := e_126_1
  simp only [e_126_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f924 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 15) = bselect (a (.virt 368)) (a (.wire 121 31)) (a (.virt 126)) := by
  have c4854 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4855 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4856 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4857 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4858 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4859 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_2 := arithEq_of_rows h (i := 2) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_2
  simp only [← c4854, ← c4855, ← c4856] at e_126_2
  have e_126_3 := arithEq_of_rows h (i := 3) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_3
  simp only [← c4857, ← c4858, ← c4859] at e_126_3
  have hr := e_126_3
  simp only [e_126_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f925 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 23) = bselect (a (.virt 368)) (a (.virt 123)) (a (.virt 127)) := by
  have c4860 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4861 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4862 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4863 := (leafCircuit_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4864 := (leafCircuit_copies152 a h).1
  have c4865 := (leafCircuit_copies152 a h).2.1
  have e_126_4 := arithEq_of_rows h (i := 4) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_4
  simp only [← c4860, ← c4861, ← c4862] at e_126_4
  have e_126_5 := arithEq_of_rows h (i := 5) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_5
  simp only [← c4863, ← c4864, ← c4865] at e_126_5
  have hr := e_126_5
  simp only [e_126_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f926 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 31) = bselect (a (.virt 370)) (a (.wire 121 7)) (a (.wire 126 23)) := by
  have c4866 := (leafCircuit_copies152 a h).2.2.1
  have c4867 := (leafCircuit_copies152 a h).2.2.2.1
  have c4868 := (leafCircuit_copies152 a h).2.2.2.2.1
  have c4869 := (leafCircuit_copies152 a h).2.2.2.2.2.1
  have c4870 := (leafCircuit_copies152 a h).2.2.2.2.2.2.1
  have c4871 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.1
  have e_126_6 := arithEq_of_rows h (i := 6) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_6
  simp only [← c4866, ← c4867, ← c4868] at e_126_6
  have e_126_7 := arithEq_of_rows h (i := 7) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_7
  simp only [← c4869, ← c4870, ← c4871] at e_126_7
  have hr := e_126_7
  simp only [e_126_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f927 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 39) = bselect (a (.virt 368)) (a (.virt 124)) (a (.virt 128)) := by
  have c4872 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.1
  have c4873 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.1
  have c4874 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4875 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4876 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4877 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_8 := arithEq_of_rows h (i := 8) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_8
  simp only [← c4872, ← c4873, ← c4874] at e_126_8
  have e_126_9 := arithEq_of_rows h (i := 9) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_9
  simp only [← c4875, ← c4876, ← c4877] at e_126_9
  have hr := e_126_9
  simp only [e_126_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f928 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 47) = bselect (a (.virt 370)) (a (.wire 121 15)) (a (.wire 126 39)) := by
  have c4878 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4879 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4880 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4881 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4882 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4883 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_10 := arithEq_of_rows h (i := 10) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_10
  simp only [← c4878, ← c4879, ← c4880] at e_126_10
  have e_126_11 := arithEq_of_rows h (i := 11) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_11
  simp only [← c4881, ← c4882, ← c4883] at e_126_11
  have hr := e_126_11
  simp only [e_126_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f929 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 126 55) = bselect (a (.virt 368)) (a (.virt 125)) (a (.virt 129)) := by
  have c4884 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4885 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4886 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4887 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4888 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4889 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_12 := arithEq_of_rows h (i := 12) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_12
  simp only [← c4884, ← c4885, ← c4886] at e_126_12
  have e_126_13 := arithEq_of_rows h (i := 13) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_13
  simp only [← c4887, ← c4888, ← c4889] at e_126_13
  have hr := e_126_13
  simp only [e_126_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f930 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 3) = bselect (a (.virt 370)) (a (.wire 121 23)) (a (.wire 126 55)) := by
  have c4890 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4891 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4892 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4893 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4894 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4895 := (leafCircuit_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_126_14 := arithEq_of_rows h (i := 14) leafCircuit_row126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_14
  simp only [← c4890, ← c4891, ← c4892] at e_126_14
  have e_127_0 := arithEq_of_rows h (i := 0) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_0
  simp only [← c4893, ← c4894, ← c4895] at e_127_0
  have hr := e_127_0
  simp only [e_126_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f931 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 11) = bselect (a (.virt 368)) (a (.virt 126)) (a (.virt 130)) := by
  have c4896 := (leafCircuit_copies153 a h).1
  have c4897 := (leafCircuit_copies153 a h).2.1
  have c4898 := (leafCircuit_copies153 a h).2.2.1
  have c4899 := (leafCircuit_copies153 a h).2.2.2.1
  have c4900 := (leafCircuit_copies153 a h).2.2.2.2.1
  have c4901 := (leafCircuit_copies153 a h).2.2.2.2.2.1
  have e_127_1 := arithEq_of_rows h (i := 1) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_1
  simp only [← c4896, ← c4897, ← c4898] at e_127_1
  have e_127_2 := arithEq_of_rows h (i := 2) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_2
  simp only [← c4899, ← c4900, ← c4901] at e_127_2
  have hr := e_127_2
  simp only [e_127_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f932 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 19) = bselect (a (.virt 370)) (a (.wire 121 31)) (a (.wire 127 11)) := by
  have c4902 := (leafCircuit_copies153 a h).2.2.2.2.2.2.1
  have c4903 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.1
  have c4904 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.1
  have c4905 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.1
  have c4906 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4907 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_3 := arithEq_of_rows h (i := 3) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_3
  simp only [← c4902, ← c4903, ← c4904] at e_127_3
  have e_127_4 := arithEq_of_rows h (i := 4) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_4
  simp only [← c4905, ← c4906, ← c4907] at e_127_4
  have hr := e_127_4
  simp only [e_127_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f933 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) := by
  have c4908 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4910 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4913 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_7 := arithEq_of_rows h (i := 7) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c4908, ← c4909, ← c4910] at e_104_7
  have e_128_0 := arithEq_of_rows h (i := 0) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_0
  simp only [← c4911, ← c4912, k3, ← c4913] at e_128_0
  have hr := e_128_0
  simp only [e_104_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f934 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 27) = bselect (a (.wire 128 3)) (a (.virt 127)) (a (.virt 131)) := by
  have c4914 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4915 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4916 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4917 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4918 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4919 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_5 := arithEq_of_rows h (i := 5) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_5
  simp only [← c4914, ← c4915, ← c4916] at e_127_5
  have e_127_6 := arithEq_of_rows h (i := 6) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_6
  simp only [← c4917, ← c4918, ← c4919] at e_127_6
  have hr := e_127_6
  simp only [e_127_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f935 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 35) = bselect (a (.virt 372)) (a (.wire 121 7)) (a (.wire 127 27)) := by
  have c4920 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4921 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4922 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4923 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4924 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4925 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_7 := arithEq_of_rows h (i := 7) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_7
  simp only [← c4920, ← c4921, ← c4922] at e_127_7
  have e_127_8 := arithEq_of_rows h (i := 8) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_8
  simp only [← c4923, ← c4924, ← c4925] at e_127_8
  have hr := e_127_8
  simp only [e_127_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f936 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) := by
  have c4908 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4910 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4913 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_7 := arithEq_of_rows h (i := 7) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c4908, ← c4909, ← c4910] at e_104_7
  have e_128_0 := arithEq_of_rows h (i := 0) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_0
  simp only [← c4911, ← c4912, k3, ← c4913] at e_128_0
  have hr := e_128_0
  simp only [e_104_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f937 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 43) = bselect (a (.wire 128 3)) (a (.virt 128)) (a (.virt 132)) := by
  have c4926 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4927 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4928 := (leafCircuit_copies154 a h).1
  have c4929 := (leafCircuit_copies154 a h).2.1
  have c4930 := (leafCircuit_copies154 a h).2.2.1
  have c4931 := (leafCircuit_copies154 a h).2.2.2.1
  have e_127_9 := arithEq_of_rows h (i := 9) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_9
  simp only [← c4926, ← c4927, ← c4928] at e_127_9
  have e_127_10 := arithEq_of_rows h (i := 10) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_10
  simp only [← c4929, ← c4930, ← c4931] at e_127_10
  have hr := e_127_10
  simp only [e_127_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f938 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 51) = bselect (a (.virt 372)) (a (.wire 121 15)) (a (.wire 127 43)) := by
  have c4932 := (leafCircuit_copies154 a h).2.2.2.2.1
  have c4933 := (leafCircuit_copies154 a h).2.2.2.2.2.1
  have c4934 := (leafCircuit_copies154 a h).2.2.2.2.2.2.1
  have c4935 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.1
  have c4936 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.1
  have c4937 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.1
  have e_127_11 := arithEq_of_rows h (i := 11) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_11
  simp only [← c4932, ← c4933, ← c4934] at e_127_11
  have e_127_12 := arithEq_of_rows h (i := 12) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_12
  simp only [← c4935, ← c4936, ← c4937] at e_127_12
  have hr := e_127_12
  simp only [e_127_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f939 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) := by
  have c4908 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4910 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4913 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_7 := arithEq_of_rows h (i := 7) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c4908, ← c4909, ← c4910] at e_104_7
  have e_128_0 := arithEq_of_rows h (i := 0) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_0
  simp only [← c4911, ← c4912, k3, ← c4913] at e_128_0
  have hr := e_128_0
  simp only [e_104_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f940 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 127 59) = bselect (a (.wire 128 3)) (a (.virt 129)) (a (.virt 133)) := by
  have c4938 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4939 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4940 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4941 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4942 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4943 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_13 := arithEq_of_rows h (i := 13) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_13
  simp only [← c4938, ← c4939, ← c4940] at e_127_13
  have e_127_14 := arithEq_of_rows h (i := 14) leafCircuit_row127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_14
  simp only [← c4941, ← c4942, ← c4943] at e_127_14
  have hr := e_127_14
  simp only [e_127_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f941 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 7) = bselect (a (.virt 372)) (a (.wire 121 23)) (a (.wire 127 59)) := by
  have c4944 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4945 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4946 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4947 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4948 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4949 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_0 := arithEq_of_rows h (i := 0) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_0
  simp only [← c4944, ← c4945, ← c4946] at e_129_0
  have e_129_1 := arithEq_of_rows h (i := 1) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_1
  simp only [← c4947, ← c4948, ← c4949] at e_129_1
  have hr := e_129_1
  simp only [e_129_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f942 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) := by
  have c4908 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4910 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4913 := (leafCircuit_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_7 := arithEq_of_rows h (i := 7) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c4908, ← c4909, ← c4910] at e_104_7
  have e_128_0 := arithEq_of_rows h (i := 0) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_0
  simp only [← c4911, ← c4912, k3, ← c4913] at e_128_0
  have hr := e_128_0
  simp only [e_104_7] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f943 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 15) = bselect (a (.wire 128 3)) (a (.virt 130)) (a (.virt 134)) := by
  have c4950 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4951 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4952 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4953 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4954 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4955 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_2 := arithEq_of_rows h (i := 2) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_2
  simp only [← c4950, ← c4951, ← c4952] at e_129_2
  have e_129_3 := arithEq_of_rows h (i := 3) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_3
  simp only [← c4953, ← c4954, ← c4955] at e_129_3
  have hr := e_129_3
  simp only [e_129_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f944 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 23) = bselect (a (.virt 372)) (a (.wire 121 31)) (a (.wire 129 15)) := by
  have c4956 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4957 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4958 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4959 := (leafCircuit_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4960 := (leafCircuit_copies155 a h).1
  have c4961 := (leafCircuit_copies155 a h).2.1
  have e_129_4 := arithEq_of_rows h (i := 4) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_4
  simp only [← c4956, ← c4957, ← c4958] at e_129_4
  have e_129_5 := arithEq_of_rows h (i := 5) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_5
  simp only [← c4959, ← c4960, ← c4961] at e_129_5
  have hr := e_129_5
  simp only [e_129_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f945 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 31) = bselect (a (.virt 374)) (a (.wire 121 7)) (a (.virt 131)) := by
  have c4962 := (leafCircuit_copies155 a h).2.2.1
  have c4963 := (leafCircuit_copies155 a h).2.2.2.1
  have c4964 := (leafCircuit_copies155 a h).2.2.2.2.1
  have c4965 := (leafCircuit_copies155 a h).2.2.2.2.2.1
  have c4966 := (leafCircuit_copies155 a h).2.2.2.2.2.2.1
  have c4967 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.1
  have e_129_6 := arithEq_of_rows h (i := 6) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_6
  simp only [← c4962, ← c4963, ← c4964] at e_129_6
  have e_129_7 := arithEq_of_rows h (i := 7) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_7
  simp only [← c4965, ← c4966, ← c4967] at e_129_7
  have hr := e_129_7
  simp only [e_129_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f946 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 39) = bselect (a (.virt 374)) (a (.wire 121 15)) (a (.virt 132)) := by
  have c4968 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.1
  have c4969 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.1
  have c4970 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4971 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4972 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4973 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_8 := arithEq_of_rows h (i := 8) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_8
  simp only [← c4968, ← c4969, ← c4970] at e_129_8
  have e_129_9 := arithEq_of_rows h (i := 9) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_9
  simp only [← c4971, ← c4972, ← c4973] at e_129_9
  have hr := e_129_9
  simp only [e_129_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f947 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 47) = bselect (a (.virt 374)) (a (.wire 121 23)) (a (.virt 133)) := by
  have c4974 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4975 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4976 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4977 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4978 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4979 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_10 := arithEq_of_rows h (i := 10) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_10
  simp only [← c4974, ← c4975, ← c4976] at e_129_10
  have e_129_11 := arithEq_of_rows h (i := 11) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_11
  simp only [← c4977, ← c4978, ← c4979] at e_129_11
  have hr := e_129_11
  simp only [e_129_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f948 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 129 55) = bselect (a (.virt 374)) (a (.wire 121 31)) (a (.virt 134)) := by
  have c4980 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4981 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4982 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4983 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4984 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4985 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_12 := arithEq_of_rows h (i := 12) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_12
  simp only [← c4980, ← c4981, ← c4982] at e_129_12
  have e_129_13 := arithEq_of_rows h (i := 13) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_13
  simp only [← c4983, ← c4984, ← c4985] at e_129_13
  have hr := e_129_13
  simp only [e_129_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f949 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 132 12) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 0 ∧ a (.wire 132 13) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 1 ∧ a (.wire 132 14) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 2 ∧ a (.wire 132 15) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 3) := by
  have c4986 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4987 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4988 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4989 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4990 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4991 := (leafCircuit_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4992 := (leafCircuit_copies156 a h).1
  have c4993 := (leafCircuit_copies156 a h).2.1
  have c4994 := (leafCircuit_copies156 a h).2.2.1
  have c4995 := (leafCircuit_copies156 a h).2.2.2.1
  have c4996 := (leafCircuit_copies156 a h).2.2.2.2.1
  have c4997 := (leafCircuit_copies156 a h).2.2.2.2.2.1
  have c4998 := (leafCircuit_copies156 a h).2.2.2.2.2.2.1
  have c4999 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.1
  have c5000 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.1
  have c5001 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.1
  have c5002 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5003 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5004 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5005 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5006 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5007 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5008 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5009 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5010 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5011 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5012 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5013 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5014 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5015 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5016 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5017 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5018 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5019 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5020 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5021 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5022 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5023 := (leafCircuit_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5024 := (leafCircuit_copies157 a h).1
  have c5025 := (leafCircuit_copies157 a h).2.1
  have c5026 := (leafCircuit_copies157 a h).2.2.1
  have c5027 := (leafCircuit_copies157 a h).2.2.2.1
  have c5028 := (leafCircuit_copies157 a h).2.2.2.2.1
  have c5029 := (leafCircuit_copies157 a h).2.2.2.2.2.1
  have c5030 := (leafCircuit_copies157 a h).2.2.2.2.2.2.1
  have c5031 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.1
  have c5032 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.1
  have c5033 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.1
  have c5034 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5035 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5036 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5037 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5038 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5039 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5040 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5041 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5042 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5043 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5044 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5045 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5046 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5047 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5048 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_128_1 := arithEq_of_rows h (i := 1) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_1
  simp only [← c4998, ← c4999, k3, ← c5000] at e_128_1
  have e_128_2 := arithEq_of_rows h (i := 2) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_2
  simp only [← c5001, ← c5002, k3, ← c5003] at e_128_2
  have e_128_3 := arithEq_of_rows h (i := 3) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_3
  simp only [← c5004, ← c5005, k3, ← c5006] at e_128_3
  have e_128_4 := arithEq_of_rows h (i := 4) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_4
  simp only [← c5007, ← c5008, k3, ← c5009] at e_128_4
  have e_128_5 := arithEq_of_rows h (i := 5) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_5
  simp only [← c5010, ← c5011, k3, ← c5012] at e_128_5
  have e_128_6 := arithEq_of_rows h (i := 6) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_6
  simp only [← c5013, ← c5014, k3, ← c5015] at e_128_6
  have e_128_7 := arithEq_of_rows h (i := 7) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_7
  simp only [← c5016, ← c5017, k3, ← c5018] at e_128_7
  have e_128_8 := arithEq_of_rows h (i := 8) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_8
  simp only [← c5019, ← c5020, k3, ← c5021] at e_128_8
  have e_128_9 := arithEq_of_rows h (i := 9) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_9
  simp only [← c5034, ← c5035, k3, ← c5036, k3] at e_128_9
  have hin0 : poseidon2In a 130 = addBlock (fun _ => 0) [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19)] :=
    poseidon2In_first c4986.symm c4987.symm c4988.symm c4989.symm c4990.symm c4991.symm c4992.symm c4993.symm (c4994.symm.trans k0) (c4995.symm.trans k0) (c4996.symm.trans k0) (c4997.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 130) rfl rfl hin0
  have hin1 : poseidon2In a 131 = addBlock (poseidon2Out a 130) [a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] :=
    poseidon2In_chain (by linear_combination -c5022 + e_128_1) (by linear_combination -c5023 + e_128_2) (by linear_combination -c5024 + e_128_3) (by linear_combination -c5025 + e_128_4) (by linear_combination -c5026 + e_128_5) (by linear_combination -c5027 + e_128_6) (by linear_combination -c5028 + e_128_7) (by linear_combination -c5029 + e_128_8) c5030.symm c5031.symm c5032.symm c5033.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 131) rfl rfl hin1
  have hin2 : poseidon2In a 132 = addBlock (poseidon2Out a 131) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c5037 + e_128_9) (c5038.symm.trans (add_zero _).symm) (c5039.symm.trans (add_zero _).symm) (c5040.symm.trans (add_zero _).symm) (c5041.symm.trans (add_zero _).symm) (c5042.symm.trans (add_zero _).symm) (c5043.symm.trans (add_zero _).symm) (c5044.symm.trans (add_zero _).symm) c5045.symm c5046.symm c5047.symm c5048.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 132) rfl rfl hin2
  have hpad : pad10 [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] =
      [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19)] ++ ([a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f950 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 3) = bselect (a (.wire 111 55)) (a (.wire 132 12)) (a (.wire 121 7)) := by
  have c5049 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5050 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5051 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5052 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5053 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5054 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_14 := arithEq_of_rows h (i := 14) leafCircuit_row129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_14
  simp only [← c5049, ← c5050, ← c5051] at e_129_14
  have e_133_0 := arithEq_of_rows h (i := 0) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_0
  simp only [← c5052, ← c5053, ← c5054] at e_133_0
  have hr := e_133_0
  simp only [e_129_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f951 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 11) = bselect (a (.wire 111 55)) (a (.wire 132 13)) (a (.wire 121 15)) := by
  have c5055 := (leafCircuit_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5056 := (leafCircuit_copies158 a h).1
  have c5057 := (leafCircuit_copies158 a h).2.1
  have c5058 := (leafCircuit_copies158 a h).2.2.1
  have c5059 := (leafCircuit_copies158 a h).2.2.2.1
  have c5060 := (leafCircuit_copies158 a h).2.2.2.2.1
  have e_133_1 := arithEq_of_rows h (i := 1) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_1
  simp only [← c5055, ← c5056, ← c5057] at e_133_1
  have e_133_2 := arithEq_of_rows h (i := 2) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_2
  simp only [← c5058, ← c5059, ← c5060] at e_133_2
  have hr := e_133_2
  simp only [e_133_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f952 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 19) = bselect (a (.wire 111 55)) (a (.wire 132 14)) (a (.wire 121 23)) := by
  have c5061 := (leafCircuit_copies158 a h).2.2.2.2.2.1
  have c5062 := (leafCircuit_copies158 a h).2.2.2.2.2.2.1
  have c5063 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.1
  have c5064 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.1
  have c5065 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.1
  have c5066 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_133_3 := arithEq_of_rows h (i := 3) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_3
  simp only [← c5061, ← c5062, ← c5063] at e_133_3
  have e_133_4 := arithEq_of_rows h (i := 4) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_4
  simp only [← c5064, ← c5065, ← c5066] at e_133_4
  have hr := e_133_4
  simp only [e_133_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f953 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 27) = bselect (a (.wire 111 55)) (a (.wire 132 15)) (a (.wire 121 31)) := by
  have c5067 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5068 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5069 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5070 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5071 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5072 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_5 := arithEq_of_rows h (i := 5) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_5
  simp only [← c5067, ← c5068, ← c5069] at e_133_5
  have e_133_6 := arithEq_of_rows h (i := 6) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_6
  simp only [← c5070, ← c5071, ← c5072] at e_133_6
  have hr := e_133_6
  simp only [e_133_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f954 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 134 1), a (.wire 134 2), a (.wire 134 3), a (.wire 134 4), a (.wire 134 5)] := by
  have c5073 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5074 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5075 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5076 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5077 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5078 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5079 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5080 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5081 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5082 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5083 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5084 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5085 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5086 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5087 := (leafCircuit_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5088 := (leafCircuit_copies159 a h).1
  have c5089 := (leafCircuit_copies159 a h).2.1
  have c5090 := (leafCircuit_copies159 a h).2.2.1
  have c5091 := (leafCircuit_copies159 a h).2.2.2.1
  have c5092 := (leafCircuit_copies159 a h).2.2.2.2.1
  have c5093 := (leafCircuit_copies159 a h).2.2.2.2.2.1
  have c5094 := (leafCircuit_copies159 a h).2.2.2.2.2.2.1
  have c5095 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.1
  have c5096 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.1
  have c5097 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.1
  have c5098 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5099 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5100 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5101 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5102 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5103 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5104 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5105 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5106 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5107 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5108 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5109 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5110 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5111 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5112 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5113 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5114 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5115 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5116 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5117 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5118 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5119 := (leafCircuit_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5120 := (leafCircuit_copies160 a h).1
  have c5121 := (leafCircuit_copies160 a h).2.1
  have c5122 := (leafCircuit_copies160 a h).2.2.1
  have c5123 := (leafCircuit_copies160 a h).2.2.2.1
  have c5124 := (leafCircuit_copies160 a h).2.2.2.2.1
  have c5125 := (leafCircuit_copies160 a h).2.2.2.2.2.1
  have c5126 := (leafCircuit_copies160 a h).2.2.2.2.2.2.1
  have c5127 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 134) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c5073.trans k0
    · exact c5074.trans k0
    · exact c5075.trans k0
    · exact c5076.trans k0
    · exact c5077.trans k0
    · exact c5078.trans k0
    · exact c5079.trans k0
    · exact c5080.trans k0
    · exact c5081.trans k0
    · exact c5082.trans k0
    · exact c5083.trans k0
    · exact c5084.trans k0
    · exact c5085.trans k0
    · exact c5086.trans k0
    · exact c5087.trans k0
    · exact c5088.trans k0
    · exact c5089.trans k0
    · exact c5090.trans k0
    · exact c5091.trans k0
    · exact c5092.trans k0
    · exact c5093.trans k0
    · exact c5094.trans k0
    · exact c5095.trans k0
    · exact c5096.trans k0
    · exact c5097.trans k0
    · exact c5098.trans k0
    · exact c5099.trans k0
    · exact c5100.trans k0
    · exact c5101.trans k0
    · exact c5102.trans k0
    · exact c5103.trans k0
    · exact c5104.trans k0
    · exact c5105.trans k0
    · exact c5106.trans k0
    · exact c5107.trans k0
    · exact c5108.trans k0
    · exact c5109.trans k0
    · exact c5110.trans k0
    · exact c5111.trans k0
    · exact c5112.trans k0
    · exact c5113.trans k0
    · exact c5114.trans k0
    · exact c5115.trans k0
    · exact c5116.trans k0
    · exact c5117.trans k0
    · exact c5118.trans k0
    · exact c5119.trans k0
    · exact c5120.trans k0
    · exact c5121.trans k0
    · exact c5122.trans k0
    · exact c5123.trans k0
    · exact c5124.trans k0
    · exact c5125.trans k0
    · exact c5126.trans k0
    )
  rwa [c5127] at hr

theorem leafCircuit_f955 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f956 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 5) = band (a (.virt 292)) (a (.wire 134 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f957 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 5) = band (a (.wire 134 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f958 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 5) = bor (a (.virt 14)) (a (.wire 134 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f959 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 134 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f960 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f961 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 5) = a (.virt 14) + a (.wire 134 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f962 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 5) = a (.wire 134 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f963 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 31) = bnot (a (.wire 134 5)) := by
  have c5128 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.1
  have c5129 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.1
  have c5130 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_7 := arithEq_of_rows h (i := 7) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_7
  simp only [← c5128, k3, ← c5129, k3, ← c5130] at e_133_7
  have hr := e_133_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f964 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 31) = band (a (.virt 292)) (a (.wire 133 31)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f965 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f966 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 134 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f967 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 133 31)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f968 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 5) = bor (a (.wire 134 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f969 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 4) = a (.virt 292) * a (.wire 134 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f970 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 19) = a (.virt 294) * a (.wire 134 4) := by
  have c5131 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5132 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5133 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_125_4 := arithEq_of_rows h (i := 4) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_4
  simp only [← c5131, k5, ← c5132, ← c5133, k5] at e_125_4
  have hr := e_125_4
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f971 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 43) = a (.virt 292) + a (.wire 134 4) := by
  have c5134 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5135 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5136 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_128_10 := arithEq_of_rows h (i := 10) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_10
  simp only [← c5134, k3, ← c5135, k3, ← c5136] at e_128_10
  have hr := e_128_10
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f972 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 35) = a (.wire 128 43) - a (.wire 125 19) := by
  have c5137 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5138 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5139 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_8 := arithEq_of_rows h (i := 8) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_8
  simp only [← c5137, ← c5138, k3, ← c5139] at e_133_8
  have hr := e_133_8
  linear_combination hr

theorem leafCircuit_f973 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 39) = bnot (a (.wire 133 35)) := by
  have c5140 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5141 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5142 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_9 := arithEq_of_rows h (i := 9) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_9
  simp only [← c5140, k3, ← c5141, k3, ← c5142] at e_133_9
  have hr := e_133_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f974 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 23) = band (a (.wire 133 31)) (a (.wire 133 39)) := by
  have c5143 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5144 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5145 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_5 := arithEq_of_rows h (i := 5) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_5
  simp only [← c5143, ← c5144, ← c5145] at e_125_5
  have hr := e_125_5
  simp only [band]
  linear_combination hr

theorem leafCircuit_f975 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f976 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 3) = band (a (.virt 292)) (a (.wire 134 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f977 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 27) = band (a (.wire 134 3)) (a (.wire 125 23)) := by
  have c5146 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5147 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5148 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_6 := arithEq_of_rows h (i := 6) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_6
  simp only [← c5146, ← c5147, ← c5148] at e_125_6
  have hr := e_125_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f978 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 47) = bor (a (.wire 134 5)) (a (.wire 125 27)) := by
  have c5149 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5150 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5151 := (leafCircuit_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5152 := (leafCircuit_copies161 a h).1
  have c5153 := (leafCircuit_copies161 a h).2.1
  have c5154 := (leafCircuit_copies161 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_8 := arithEq_of_rows h (i := 8) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_8
  simp only [← c5149, ← c5150, ← c5151] at e_104_8
  have e_128_11 := arithEq_of_rows h (i := 11) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_11
  simp only [← c5152, ← c5153, k3, ← c5154] at e_128_11
  have hr := e_128_11
  simp only [e_104_8] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f979 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 134 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f980 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f981 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 3) = a (.virt 14) + a (.wire 134 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f982 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 3) = a (.wire 134 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f983 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 43) = bnot (a (.wire 134 3)) := by
  have c5155 := (leafCircuit_copies161 a h).2.2.2.1
  have c5156 := (leafCircuit_copies161 a h).2.2.2.2.1
  have c5157 := (leafCircuit_copies161 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_10 := arithEq_of_rows h (i := 10) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_10
  simp only [← c5155, k3, ← c5156, k3, ← c5157] at e_133_10
  have hr := e_133_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f984 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 31) = band (a (.wire 125 23)) (a (.wire 133 43)) := by
  have c5158 := (leafCircuit_copies161 a h).2.2.2.2.2.2.1
  have c5159 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.1
  have c5160 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.1
  have e_125_7 := arithEq_of_rows h (i := 7) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_7
  simp only [← c5158, ← c5159, ← c5160] at e_125_7
  have hr := e_125_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f985 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f986 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 134 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f987 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 125 31)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f988 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 47) = bor (a (.wire 128 47)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f989 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 2) = a (.virt 292) * a (.wire 134 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f990 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 35) = a (.virt 294) * a (.wire 134 2) := by
  have c5161 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.1
  have c5162 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5163 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_125_8 := arithEq_of_rows h (i := 8) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_8
  simp only [← c5161, k5, ← c5162, ← c5163, k5] at e_125_8
  have hr := e_125_8
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f991 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 51) = a (.virt 292) + a (.wire 134 2) := by
  have c5164 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5165 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5166 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_128_12 := arithEq_of_rows h (i := 12) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_12
  simp only [← c5164, k3, ← c5165, k3, ← c5166] at e_128_12
  have hr := e_128_12
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f992 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 47) = a (.wire 128 51) - a (.wire 125 35) := by
  have c5167 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5168 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5169 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_11 := arithEq_of_rows h (i := 11) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_11
  simp only [← c5167, ← c5168, k3, ← c5169] at e_133_11
  have hr := e_133_11
  linear_combination hr

theorem leafCircuit_f993 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 51) = bnot (a (.wire 133 47)) := by
  have c5170 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5171 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5172 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_12 := arithEq_of_rows h (i := 12) leafCircuit_row133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_12
  simp only [← c5170, k3, ← c5171, k3, ← c5172] at e_133_12
  have hr := e_133_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f994 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 39) = band (a (.wire 125 31)) (a (.wire 133 51)) := by
  have c5173 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5174 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5175 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_9 := arithEq_of_rows h (i := 9) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_9
  simp only [← c5173, ← c5174, ← c5175] at e_125_9
  have hr := e_125_9
  simp only [band]
  linear_combination hr

theorem leafCircuit_f995 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f996 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 1) = band (a (.virt 292)) (a (.wire 134 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f997 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 43) = band (a (.wire 134 1)) (a (.wire 125 39)) := by
  have c5176 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5177 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5178 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_10 := arithEq_of_rows h (i := 10) leafCircuit_row125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_10
  simp only [← c5176, ← c5177, ← c5178] at e_125_10
  have hr := e_125_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f998 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 55) = bor (a (.wire 128 47)) (a (.wire 125 43)) := by
  have c5179 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5180 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5181 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5182 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5183 := (leafCircuit_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5184 := (leafCircuit_copies162 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_9 := arithEq_of_rows h (i := 9) leafCircuit_row104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_9
  simp only [← c5179, ← c5180, ← c5181] at e_104_9
  have e_128_13 := arithEq_of_rows h (i := 13) leafCircuit_row128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_13
  simp only [← c5182, ← c5183, k3, ← c5184] at e_128_13
  have hr := e_128_13
  simp only [e_104_9] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f999 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 134 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

end Plonky2Spec.Generated
