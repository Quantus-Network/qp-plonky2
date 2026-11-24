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

theorem leafCircuit_f1000 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1001 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 1) = a (.virt 14) + a (.wire 134 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1002 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 134 1) = a (.wire 134 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1003 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 133 55) = bnot (a (.wire 134 1)) := by
  have c5185 := (leafCircuit_copies162 a h).2.1
  have c5186 := (leafCircuit_copies162 a h).2.2.1
  have c5187 := (leafCircuit_copies162 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_133_13 := arithEq_of_rows h (row := 133) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_13
  simp only [← c5185, k3, ← c5186, k3, ← c5187] at e_133_13
  have hr := e_133_13
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1004 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 125 47) = band (a (.wire 125 39)) (a (.wire 133 55)) := by
  have c5188 := (leafCircuit_copies162 a h).2.2.2.2.1
  have c5189 := (leafCircuit_copies162 a h).2.2.2.2.2.1
  have c5190 := (leafCircuit_copies162 a h).2.2.2.2.2.2.1
  have e_125_11 := arithEq_of_rows h (row := 125) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_11
  simp only [← c5188, ← c5189, ← c5190] at e_125_11
  have hr := e_125_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1005 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 217)) 2 := by
  have c5191 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.1
  have c5192 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.1
  have c5193 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.1
  have c5194 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5195 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5196 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5197 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5198 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5199 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5200 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5201 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5202 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5203 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5204 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5205 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5206 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5207 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5208 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5209 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5210 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5211 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5212 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5213 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5214 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5215 := (leafCircuit_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5216 := (leafCircuit_copies163 a h).1
  have c5217 := (leafCircuit_copies163 a h).2.1
  have c5218 := (leafCircuit_copies163 a h).2.2.1
  have c5219 := (leafCircuit_copies163 a h).2.2.2.1
  have c5220 := (leafCircuit_copies163 a h).2.2.2.2.1
  have c5221 := (leafCircuit_copies163 a h).2.2.2.2.2.1
  have c5222 := (leafCircuit_copies163 a h).2.2.2.2.2.2.1
  have c5223 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.1
  have c5224 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.1
  have c5225 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.1
  have c5226 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5227 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5228 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5229 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5230 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5231 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5232 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5233 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5234 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5235 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5236 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5237 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5238 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5239 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5240 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5241 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5242 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5243 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5244 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5245 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5246 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5247 := (leafCircuit_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5248 := (leafCircuit_copies164 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 135) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c5191.trans k0
    · exact c5192.trans k0
    · exact c5193.trans k0
    · exact c5194.trans k0
    · exact c5195.trans k0
    · exact c5196.trans k0
    · exact c5197.trans k0
    · exact c5198.trans k0
    · exact c5199.trans k0
    · exact c5200.trans k0
    · exact c5201.trans k0
    · exact c5202.trans k0
    · exact c5203.trans k0
    · exact c5204.trans k0
    · exact c5205.trans k0
    · exact c5206.trans k0
    · exact c5207.trans k0
    · exact c5208.trans k0
    · exact c5209.trans k0
    · exact c5210.trans k0
    · exact c5211.trans k0
    · exact c5212.trans k0
    · exact c5213.trans k0
    · exact c5214.trans k0
    · exact c5215.trans k0
    · exact c5216.trans k0
    · exact c5217.trans k0
    · exact c5218.trans k0
    · exact c5219.trans k0
    · exact c5220.trans k0
    · exact c5221.trans k0
    · exact c5222.trans k0
    · exact c5223.trans k0
    · exact c5224.trans k0
    · exact c5225.trans k0
    · exact c5226.trans k0
    · exact c5227.trans k0
    · exact c5228.trans k0
    · exact c5229.trans k0
    · exact c5230.trans k0
    · exact c5231.trans k0
    · exact c5232.trans k0
    · exact c5233.trans k0
    · exact c5234.trans k0
    · exact c5235.trans k0
    · exact c5236.trans k0
    · exact c5237.trans k0
    · exact c5238.trans k0
    · exact c5239.trans k0
    · exact c5240.trans k0
    · exact c5241.trans k0
    · exact c5242.trans k0
    · exact c5243.trans k0
    · exact c5244.trans k0
    · exact c5245.trans k0
    · exact c5246.trans k0
    · exact c5247.trans k0
    )
  rwa [c5248] at hr

theorem leafCircuit_f1006 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 217)) (a (.virt 14)) (a (.virt 376)) (a (.virt 377)) := by
  have c5249 := (leafCircuit_copies164 a h).2.1
  have c5250 := (leafCircuit_copies164 a h).2.2.1
  have c5251 := (leafCircuit_copies164 a h).2.2.2.1
  have c5252 := (leafCircuit_copies164 a h).2.2.2.2.1
  have c5253 := (leafCircuit_copies164 a h).2.2.2.2.2.1
  have c5254 := (leafCircuit_copies164 a h).2.2.2.2.2.2.1
  have c5255 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.1
  have c5256 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.1
  have c5257 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.1
  have c5258 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5259 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5260 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5261 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5262 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_125_12 := arithEq_of_rows h (row := 125) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_12
  simp only [← c5252, ← c5253, ← c5254] at e_125_12
  have e_125_13 := arithEq_of_rows h (row := 125) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_13
  simp only [← c5255, ← c5256, ← c5257] at e_125_13
  have e_133_14 := arithEq_of_rows h (row := 133) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_14
  simp only [← c5249, k3, ← c5250, k3, ← c5251] at e_133_14
  have e_136_0 := arithEq_of_rows h (row := 136) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_0
  simp only [← c5258, ← c5259, k3, ← c5260] at e_136_0
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_125_12
    linear_combination c5261.trans k0 - hc
  · have hc := e_136_0
    simp only [e_133_14, e_125_13] at hc
    linear_combination c5262.trans k0 - hc

theorem leafCircuit_f1007 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 217)) (a (.virt 292)) (a (.virt 378)) (a (.virt 379)) := by
  have c5263 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5264 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5265 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5266 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5267 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5268 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5269 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5270 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5271 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5272 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5273 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5274 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5275 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5276 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5277 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5278 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5279 := (leafCircuit_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_125_14 := arithEq_of_rows h (row := 125) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_14
  simp only [← c5269, ← c5270, ← c5271] at e_125_14
  have e_136_1 := arithEq_of_rows h (row := 136) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_1
  simp only [← c5263, k3, ← c5264, k3, ← c5265] at e_136_1
  have e_136_2 := arithEq_of_rows h (row := 136) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_2
  simp only [← c5266, ← c5267, k3, ← c5268, k3] at e_136_2
  have e_136_3 := arithEq_of_rows h (row := 136) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_3
  simp only [← c5275, ← c5276, k3, ← c5277] at e_136_3
  have e_137_0 := arithEq_of_rows h (row := 137) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_0
  simp only [← c5272, ← c5273, ← c5274] at e_137_0
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_125_14
    simp only [e_136_2] at hc
    linear_combination c5278.trans k0 - hc
  · have hc := e_136_3
    simp only [e_136_1, e_137_0, e_136_2] at hc
    linear_combination c5279.trans k0 - hc

theorem leafCircuit_f1008 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 217)) (a (.virt 294)) (a (.virt 380)) (a (.virt 381)) := by
  have c5280 := (leafCircuit_copies165 a h).1
  have c5281 := (leafCircuit_copies165 a h).2.1
  have c5282 := (leafCircuit_copies165 a h).2.2.1
  have c5283 := (leafCircuit_copies165 a h).2.2.2.1
  have c5284 := (leafCircuit_copies165 a h).2.2.2.2.1
  have c5285 := (leafCircuit_copies165 a h).2.2.2.2.2.1
  have c5286 := (leafCircuit_copies165 a h).2.2.2.2.2.2.1
  have c5287 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.1
  have c5288 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.1
  have c5289 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.1
  have c5290 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5291 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5292 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5293 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5294 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5295 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5296 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_136_4 := arithEq_of_rows h (row := 136) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_4
  simp only [← c5280, k3, ← c5281, k3, ← c5282] at e_136_4
  have e_136_5 := arithEq_of_rows h (row := 136) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_5
  simp only [← c5283, ← c5284, k3, ← c5285, k5] at e_136_5
  have e_136_6 := arithEq_of_rows h (row := 136) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_6
  simp only [← c5292, ← c5293, k3, ← c5294] at e_136_6
  have e_137_1 := arithEq_of_rows h (row := 137) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_1
  simp only [← c5286, ← c5287, ← c5288] at e_137_1
  have e_137_2 := arithEq_of_rows h (row := 137) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_2
  simp only [← c5289, ← c5290, ← c5291] at e_137_2
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_137_1
    simp only [e_136_5] at hc
    linear_combination c5295.trans k0 - hc
  · have hc := e_136_6
    simp only [e_136_4, e_137_2, e_136_5] at hc
    linear_combination c5296.trans k0 - hc

theorem leafCircuit_f1009 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 217)) (a (.virt 295)) (a (.virt 382)) (a (.virt 383)) := by
  have c5297 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5298 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5299 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5300 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5301 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5302 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5303 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5304 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5305 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5306 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5307 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5308 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5309 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5310 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5311 := (leafCircuit_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5312 := (leafCircuit_copies166 a h).1
  have c5313 := (leafCircuit_copies166 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_136_7 := arithEq_of_rows h (row := 136) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_7
  simp only [← c5297, k3, ← c5298, k3, ← c5299] at e_136_7
  have e_136_8 := arithEq_of_rows h (row := 136) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_8
  simp only [← c5300, ← c5301, k3, ← c5302, k6] at e_136_8
  have e_136_9 := arithEq_of_rows h (row := 136) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_9
  simp only [← c5309, ← c5310, k3, ← c5311] at e_136_9
  have e_137_3 := arithEq_of_rows h (row := 137) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_3
  simp only [← c5303, ← c5304, ← c5305] at e_137_3
  have e_137_4 := arithEq_of_rows h (row := 137) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_4
  simp only [← c5306, ← c5307, ← c5308] at e_137_4
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_137_3
    simp only [e_136_8] at hc
    linear_combination c5312.trans k0 - hc
  · have hc := e_136_9
    simp only [e_136_7, e_137_4, e_136_8] at hc
    linear_combination c5313.trans k0 - hc

theorem leafCircuit_f1010 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 136 47) = bselect (a (.virt 376)) (a (.wire 133 3)) (a (.virt 135)) := by
  have c5314 := (leafCircuit_copies166 a h).2.2.1
  have c5315 := (leafCircuit_copies166 a h).2.2.2.1
  have c5316 := (leafCircuit_copies166 a h).2.2.2.2.1
  have c5317 := (leafCircuit_copies166 a h).2.2.2.2.2.1
  have c5318 := (leafCircuit_copies166 a h).2.2.2.2.2.2.1
  have c5319 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.1
  have e_136_10 := arithEq_of_rows h (row := 136) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_10
  simp only [← c5314, ← c5315, ← c5316] at e_136_10
  have e_136_11 := arithEq_of_rows h (row := 136) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_11
  simp only [← c5317, ← c5318, ← c5319] at e_136_11
  have hr := e_136_11
  simp only [e_136_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1011 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 136 55) = bselect (a (.virt 376)) (a (.wire 133 11)) (a (.virt 136)) := by
  have c5320 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.1
  have c5321 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.1
  have c5322 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5323 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5324 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5325 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_12 := arithEq_of_rows h (row := 136) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_12
  simp only [← c5320, ← c5321, ← c5322] at e_136_12
  have e_136_13 := arithEq_of_rows h (row := 136) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_13
  simp only [← c5323, ← c5324, ← c5325] at e_136_13
  have hr := e_136_13
  simp only [e_136_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1012 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 3) = bselect (a (.virt 376)) (a (.wire 133 19)) (a (.virt 137)) := by
  have c5326 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5327 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5328 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5329 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5330 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5331 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_14 := arithEq_of_rows h (row := 136) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_14
  simp only [← c5326, ← c5327, ← c5328] at e_136_14
  have e_138_0 := arithEq_of_rows h (row := 138) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_0
  simp only [← c5329, ← c5330, ← c5331] at e_138_0
  have hr := e_138_0
  simp only [e_136_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1013 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 11) = bselect (a (.virt 376)) (a (.wire 133 27)) (a (.virt 138)) := by
  have c5332 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5333 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5334 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5335 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5336 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5337 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_1 := arithEq_of_rows h (row := 138) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_1
  simp only [← c5332, ← c5333, ← c5334] at e_138_1
  have e_138_2 := arithEq_of_rows h (row := 138) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_2
  simp only [← c5335, ← c5336, ← c5337] at e_138_2
  have hr := e_138_2
  simp only [e_138_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1014 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 19) = bselect (a (.virt 376)) (a (.virt 135)) (a (.virt 139)) := by
  have c5338 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5339 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5340 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5341 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5342 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5343 := (leafCircuit_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_138_3 := arithEq_of_rows h (row := 138) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_3
  simp only [← c5338, ← c5339, ← c5340] at e_138_3
  have e_138_4 := arithEq_of_rows h (row := 138) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_4
  simp only [← c5341, ← c5342, ← c5343] at e_138_4
  have hr := e_138_4
  simp only [e_138_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1015 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 27) = bselect (a (.virt 378)) (a (.wire 133 3)) (a (.wire 138 19)) := by
  have c5344 := (leafCircuit_copies167 a h).1
  have c5345 := (leafCircuit_copies167 a h).2.1
  have c5346 := (leafCircuit_copies167 a h).2.2.1
  have c5347 := (leafCircuit_copies167 a h).2.2.2.1
  have c5348 := (leafCircuit_copies167 a h).2.2.2.2.1
  have c5349 := (leafCircuit_copies167 a h).2.2.2.2.2.1
  have e_138_5 := arithEq_of_rows h (row := 138) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_5
  simp only [← c5344, ← c5345, ← c5346] at e_138_5
  have e_138_6 := arithEq_of_rows h (row := 138) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_6
  simp only [← c5347, ← c5348, ← c5349] at e_138_6
  have hr := e_138_6
  simp only [e_138_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1016 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 35) = bselect (a (.virt 376)) (a (.virt 136)) (a (.virt 140)) := by
  have c5350 := (leafCircuit_copies167 a h).2.2.2.2.2.2.1
  have c5351 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.1
  have c5352 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.1
  have c5353 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.1
  have c5354 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5355 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_7 := arithEq_of_rows h (row := 138) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_7
  simp only [← c5350, ← c5351, ← c5352] at e_138_7
  have e_138_8 := arithEq_of_rows h (row := 138) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_8
  simp only [← c5353, ← c5354, ← c5355] at e_138_8
  have hr := e_138_8
  simp only [e_138_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1017 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 43) = bselect (a (.virt 378)) (a (.wire 133 11)) (a (.wire 138 35)) := by
  have c5356 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5357 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5358 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5359 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5360 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5361 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_9 := arithEq_of_rows h (row := 138) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_9
  simp only [← c5356, ← c5357, ← c5358] at e_138_9
  have e_138_10 := arithEq_of_rows h (row := 138) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_10
  simp only [← c5359, ← c5360, ← c5361] at e_138_10
  have hr := e_138_10
  simp only [e_138_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1018 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 51) = bselect (a (.virt 376)) (a (.virt 137)) (a (.virt 141)) := by
  have c5362 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5363 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5364 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5365 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5366 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5367 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_11 := arithEq_of_rows h (row := 138) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_11
  simp only [← c5362, ← c5363, ← c5364] at e_138_11
  have e_138_12 := arithEq_of_rows h (row := 138) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_12
  simp only [← c5365, ← c5366, ← c5367] at e_138_12
  have hr := e_138_12
  simp only [e_138_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1019 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 138 59) = bselect (a (.virt 378)) (a (.wire 133 19)) (a (.wire 138 51)) := by
  have c5368 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5369 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5370 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5371 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5372 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5373 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_13 := arithEq_of_rows h (row := 138) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_13
  simp only [← c5368, ← c5369, ← c5370] at e_138_13
  have e_138_14 := arithEq_of_rows h (row := 138) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_14
  simp only [← c5371, ← c5372, ← c5373] at e_138_14
  have hr := e_138_14
  simp only [e_138_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1020 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 7) = bselect (a (.virt 376)) (a (.virt 138)) (a (.virt 142)) := by
  have c5374 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5375 := (leafCircuit_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5376 := (leafCircuit_copies168 a h).1
  have c5377 := (leafCircuit_copies168 a h).2.1
  have c5378 := (leafCircuit_copies168 a h).2.2.1
  have c5379 := (leafCircuit_copies168 a h).2.2.2.1
  have e_139_0 := arithEq_of_rows h (row := 139) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_0
  simp only [← c5374, ← c5375, ← c5376] at e_139_0
  have e_139_1 := arithEq_of_rows h (row := 139) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_1
  simp only [← c5377, ← c5378, ← c5379] at e_139_1
  have hr := e_139_1
  simp only [e_139_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1021 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 15) = bselect (a (.virt 378)) (a (.wire 133 27)) (a (.wire 139 7)) := by
  have c5380 := (leafCircuit_copies168 a h).2.2.2.2.1
  have c5381 := (leafCircuit_copies168 a h).2.2.2.2.2.1
  have c5382 := (leafCircuit_copies168 a h).2.2.2.2.2.2.1
  have c5383 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.1
  have c5384 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.1
  have c5385 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.1
  have e_139_2 := arithEq_of_rows h (row := 139) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_2
  simp only [← c5380, ← c5381, ← c5382] at e_139_2
  have e_139_3 := arithEq_of_rows h (row := 139) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_3
  simp only [← c5383, ← c5384, ← c5385] at e_139_3
  have hr := e_139_3
  simp only [e_139_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1022 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) := by
  have c5386 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5391 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c5386, ← c5387, ← c5388] at e_104_10
  have e_128_14 := arithEq_of_rows h (row := 128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_14
  simp only [← c5389, ← c5390, k3, ← c5391] at e_128_14
  have hr := e_128_14
  simp only [e_104_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1023 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 23) = bselect (a (.wire 128 59)) (a (.virt 139)) (a (.virt 143)) := by
  have c5392 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5393 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5394 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5395 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5396 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5397 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_4 := arithEq_of_rows h (row := 139) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_4
  simp only [← c5392, ← c5393, ← c5394] at e_139_4
  have e_139_5 := arithEq_of_rows h (row := 139) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_5
  simp only [← c5395, ← c5396, ← c5397] at e_139_5
  have hr := e_139_5
  simp only [e_139_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1024 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 31) = bselect (a (.virt 380)) (a (.wire 133 3)) (a (.wire 139 23)) := by
  have c5398 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5399 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5400 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5401 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5402 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5403 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_6 := arithEq_of_rows h (row := 139) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_6
  simp only [← c5398, ← c5399, ← c5400] at e_139_6
  have e_139_7 := arithEq_of_rows h (row := 139) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_7
  simp only [← c5401, ← c5402, ← c5403] at e_139_7
  have hr := e_139_7
  simp only [e_139_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1025 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) := by
  have c5386 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5391 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c5386, ← c5387, ← c5388] at e_104_10
  have e_128_14 := arithEq_of_rows h (row := 128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_14
  simp only [← c5389, ← c5390, k3, ← c5391] at e_128_14
  have hr := e_128_14
  simp only [e_104_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1026 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 39) = bselect (a (.wire 128 59)) (a (.virt 140)) (a (.virt 144)) := by
  have c5404 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5405 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5406 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5407 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5408 := (leafCircuit_copies169 a h).1
  have c5409 := (leafCircuit_copies169 a h).2.1
  have e_139_8 := arithEq_of_rows h (row := 139) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_8
  simp only [← c5404, ← c5405, ← c5406] at e_139_8
  have e_139_9 := arithEq_of_rows h (row := 139) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_9
  simp only [← c5407, ← c5408, ← c5409] at e_139_9
  have hr := e_139_9
  simp only [e_139_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1027 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 47) = bselect (a (.virt 380)) (a (.wire 133 11)) (a (.wire 139 39)) := by
  have c5410 := (leafCircuit_copies169 a h).2.2.1
  have c5411 := (leafCircuit_copies169 a h).2.2.2.1
  have c5412 := (leafCircuit_copies169 a h).2.2.2.2.1
  have c5413 := (leafCircuit_copies169 a h).2.2.2.2.2.1
  have c5414 := (leafCircuit_copies169 a h).2.2.2.2.2.2.1
  have c5415 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.1
  have e_139_10 := arithEq_of_rows h (row := 139) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_10
  simp only [← c5410, ← c5411, ← c5412] at e_139_10
  have e_139_11 := arithEq_of_rows h (row := 139) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_11
  simp only [← c5413, ← c5414, ← c5415] at e_139_11
  have hr := e_139_11
  simp only [e_139_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1028 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) := by
  have c5386 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5391 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c5386, ← c5387, ← c5388] at e_104_10
  have e_128_14 := arithEq_of_rows h (row := 128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_14
  simp only [← c5389, ← c5390, k3, ← c5391] at e_128_14
  have hr := e_128_14
  simp only [e_104_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1029 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 139 55) = bselect (a (.wire 128 59)) (a (.virt 141)) (a (.virt 145)) := by
  have c5416 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.1
  have c5417 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.1
  have c5418 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5419 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5420 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5421 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_12 := arithEq_of_rows h (row := 139) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_12
  simp only [← c5416, ← c5417, ← c5418] at e_139_12
  have e_139_13 := arithEq_of_rows h (row := 139) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_13
  simp only [← c5419, ← c5420, ← c5421] at e_139_13
  have hr := e_139_13
  simp only [e_139_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1030 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 3) = bselect (a (.virt 380)) (a (.wire 133 19)) (a (.wire 139 55)) := by
  have c5422 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5423 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5424 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5425 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5426 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5427 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_14 := arithEq_of_rows h (row := 139) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_14
  simp only [← c5422, ← c5423, ← c5424] at e_139_14
  have e_140_0 := arithEq_of_rows h (row := 140) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_0
  simp only [← c5425, ← c5426, ← c5427] at e_140_0
  have hr := e_140_0
  simp only [e_139_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1031 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) := by
  have c5386 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5391 := (leafCircuit_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c5386, ← c5387, ← c5388] at e_104_10
  have e_128_14 := arithEq_of_rows h (row := 128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_14
  simp only [← c5389, ← c5390, k3, ← c5391] at e_128_14
  have hr := e_128_14
  simp only [e_104_10] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1032 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 11) = bselect (a (.wire 128 59)) (a (.virt 142)) (a (.virt 146)) := by
  have c5428 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5429 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5430 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5431 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5432 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5433 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_1 := arithEq_of_rows h (row := 140) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_1
  simp only [← c5428, ← c5429, ← c5430] at e_140_1
  have e_140_2 := arithEq_of_rows h (row := 140) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_2
  simp only [← c5431, ← c5432, ← c5433] at e_140_2
  have hr := e_140_2
  simp only [e_140_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1033 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 19) = bselect (a (.virt 380)) (a (.wire 133 27)) (a (.wire 140 11)) := by
  have c5434 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5435 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5436 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5437 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5438 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5439 := (leafCircuit_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_140_3 := arithEq_of_rows h (row := 140) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_3
  simp only [← c5434, ← c5435, ← c5436] at e_140_3
  have e_140_4 := arithEq_of_rows h (row := 140) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_4
  simp only [← c5437, ← c5438, ← c5439] at e_140_4
  have hr := e_140_4
  simp only [e_140_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1034 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 27) = bselect (a (.virt 382)) (a (.wire 133 3)) (a (.virt 143)) := by
  have c5440 := (leafCircuit_copies170 a h).1
  have c5441 := (leafCircuit_copies170 a h).2.1
  have c5442 := (leafCircuit_copies170 a h).2.2.1
  have c5443 := (leafCircuit_copies170 a h).2.2.2.1
  have c5444 := (leafCircuit_copies170 a h).2.2.2.2.1
  have c5445 := (leafCircuit_copies170 a h).2.2.2.2.2.1
  have e_140_5 := arithEq_of_rows h (row := 140) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_5
  simp only [← c5440, ← c5441, ← c5442] at e_140_5
  have e_140_6 := arithEq_of_rows h (row := 140) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_6
  simp only [← c5443, ← c5444, ← c5445] at e_140_6
  have hr := e_140_6
  simp only [e_140_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1035 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 35) = bselect (a (.virt 382)) (a (.wire 133 11)) (a (.virt 144)) := by
  have c5446 := (leafCircuit_copies170 a h).2.2.2.2.2.2.1
  have c5447 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.1
  have c5448 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.1
  have c5449 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.1
  have c5450 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5451 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_7 := arithEq_of_rows h (row := 140) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_7
  simp only [← c5446, ← c5447, ← c5448] at e_140_7
  have e_140_8 := arithEq_of_rows h (row := 140) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_8
  simp only [← c5449, ← c5450, ← c5451] at e_140_8
  have hr := e_140_8
  simp only [e_140_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1036 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 43) = bselect (a (.virt 382)) (a (.wire 133 19)) (a (.virt 145)) := by
  have c5452 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5453 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5454 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5455 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5456 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5457 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_9 := arithEq_of_rows h (row := 140) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_9
  simp only [← c5452, ← c5453, ← c5454] at e_140_9
  have e_140_10 := arithEq_of_rows h (row := 140) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_10
  simp only [← c5455, ← c5456, ← c5457] at e_140_10
  have hr := e_140_10
  simp only [e_140_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1037 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 51) = bselect (a (.virt 382)) (a (.wire 133 27)) (a (.virt 146)) := by
  have c5458 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5459 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5460 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5461 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5462 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5463 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_11 := arithEq_of_rows h (row := 140) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_11
  simp only [← c5458, ← c5459, ← c5460] at e_140_11
  have e_140_12 := arithEq_of_rows h (row := 140) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_12
  simp only [← c5461, ← c5462, ← c5463] at e_140_12
  have hr := e_140_12
  simp only [e_140_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1038 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 144 12) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 0 ∧ a (.wire 144 13) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 1 ∧ a (.wire 144 14) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 2 ∧ a (.wire 144 15) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 3) := by
  have c5464 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5465 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5466 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5467 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5468 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5469 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5470 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5471 := (leafCircuit_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5472 := (leafCircuit_copies171 a h).1
  have c5473 := (leafCircuit_copies171 a h).2.1
  have c5474 := (leafCircuit_copies171 a h).2.2.1
  have c5475 := (leafCircuit_copies171 a h).2.2.2.1
  have c5476 := (leafCircuit_copies171 a h).2.2.2.2.1
  have c5477 := (leafCircuit_copies171 a h).2.2.2.2.2.1
  have c5478 := (leafCircuit_copies171 a h).2.2.2.2.2.2.1
  have c5479 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.1
  have c5480 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.1
  have c5481 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.1
  have c5482 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5483 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5484 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5485 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5486 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5487 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5488 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5489 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5490 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5491 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5492 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5493 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5494 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5495 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5496 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5497 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5498 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5499 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5500 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5501 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5502 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5503 := (leafCircuit_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5504 := (leafCircuit_copies172 a h).1
  have c5505 := (leafCircuit_copies172 a h).2.1
  have c5506 := (leafCircuit_copies172 a h).2.2.1
  have c5507 := (leafCircuit_copies172 a h).2.2.2.1
  have c5508 := (leafCircuit_copies172 a h).2.2.2.2.1
  have c5509 := (leafCircuit_copies172 a h).2.2.2.2.2.1
  have c5510 := (leafCircuit_copies172 a h).2.2.2.2.2.2.1
  have c5511 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.1
  have c5512 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.1
  have c5513 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.1
  have c5514 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5515 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5516 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5517 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5518 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5519 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5520 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5521 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5522 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5523 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5524 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5525 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5526 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_142_0 := arithEq_of_rows h (row := 142) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_0
  simp only [← c5476, ← c5477, k3, ← c5478] at e_142_0
  have e_142_1 := arithEq_of_rows h (row := 142) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_1
  simp only [← c5479, ← c5480, k3, ← c5481] at e_142_1
  have e_142_2 := arithEq_of_rows h (row := 142) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_2
  simp only [← c5482, ← c5483, k3, ← c5484] at e_142_2
  have e_142_3 := arithEq_of_rows h (row := 142) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_3
  simp only [← c5485, ← c5486, k3, ← c5487] at e_142_3
  have e_142_4 := arithEq_of_rows h (row := 142) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_4
  simp only [← c5488, ← c5489, k3, ← c5490] at e_142_4
  have e_142_5 := arithEq_of_rows h (row := 142) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_5
  simp only [← c5491, ← c5492, k3, ← c5493] at e_142_5
  have e_142_6 := arithEq_of_rows h (row := 142) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_6
  simp only [← c5494, ← c5495, k3, ← c5496] at e_142_6
  have e_142_7 := arithEq_of_rows h (row := 142) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_7
  simp only [← c5497, ← c5498, k3, ← c5499] at e_142_7
  have e_142_8 := arithEq_of_rows h (row := 142) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_8
  simp only [← c5512, ← c5513, k3, ← c5514, k3] at e_142_8
  have hin0 : poseidon2In a 141 = addBlock (fun _ => 0) [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15)] :=
    poseidon2In_first c5464.symm c5465.symm c5466.symm c5467.symm c5468.symm c5469.symm c5470.symm c5471.symm (c5472.symm.trans k0) (c5473.symm.trans k0) (c5474.symm.trans k0) (c5475.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 141) rfl rfl hin0
  have hin1 : poseidon2In a 143 = addBlock (poseidon2Out a 141) [a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] :=
    poseidon2In_chain (by linear_combination -c5500 + e_142_0) (by linear_combination -c5501 + e_142_1) (by linear_combination -c5502 + e_142_2) (by linear_combination -c5503 + e_142_3) (by linear_combination -c5504 + e_142_4) (by linear_combination -c5505 + e_142_5) (by linear_combination -c5506 + e_142_6) (by linear_combination -c5507 + e_142_7) c5508.symm c5509.symm c5510.symm c5511.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 143) rfl rfl hin1
  have hin2 : poseidon2In a 144 = addBlock (poseidon2Out a 143) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c5515 + e_142_8) (c5516.symm.trans (add_zero _).symm) (c5517.symm.trans (add_zero _).symm) (c5518.symm.trans (add_zero _).symm) (c5519.symm.trans (add_zero _).symm) (c5520.symm.trans (add_zero _).symm) (c5521.symm.trans (add_zero _).symm) (c5522.symm.trans (add_zero _).symm) c5523.symm c5524.symm c5525.symm c5526.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 144) rfl rfl hin2
  have hpad : pad10 [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] =
      [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15)] ++ ([a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1039 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 140 59) = bselect (a (.wire 128 55)) (a (.wire 144 12)) (a (.wire 133 3)) := by
  have c5527 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5528 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5529 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5530 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5531 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5532 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_13 := arithEq_of_rows h (row := 140) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_13
  simp only [← c5527, ← c5528, ← c5529] at e_140_13
  have e_140_14 := arithEq_of_rows h (row := 140) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_14
  simp only [← c5530, ← c5531, ← c5532] at e_140_14
  have hr := e_140_14
  simp only [e_140_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1040 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 7) = bselect (a (.wire 128 55)) (a (.wire 144 13)) (a (.wire 133 11)) := by
  have c5533 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5534 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5535 := (leafCircuit_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5536 := (leafCircuit_copies173 a h).1
  have c5537 := (leafCircuit_copies173 a h).2.1
  have c5538 := (leafCircuit_copies173 a h).2.2.1
  have e_145_0 := arithEq_of_rows h (row := 145) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_0
  simp only [← c5533, ← c5534, ← c5535] at e_145_0
  have e_145_1 := arithEq_of_rows h (row := 145) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_1
  simp only [← c5536, ← c5537, ← c5538] at e_145_1
  have hr := e_145_1
  simp only [e_145_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1041 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 15) = bselect (a (.wire 128 55)) (a (.wire 144 14)) (a (.wire 133 19)) := by
  have c5539 := (leafCircuit_copies173 a h).2.2.2.1
  have c5540 := (leafCircuit_copies173 a h).2.2.2.2.1
  have c5541 := (leafCircuit_copies173 a h).2.2.2.2.2.1
  have c5542 := (leafCircuit_copies173 a h).2.2.2.2.2.2.1
  have c5543 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.1
  have c5544 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.1
  have e_145_2 := arithEq_of_rows h (row := 145) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_2
  simp only [← c5539, ← c5540, ← c5541] at e_145_2
  have e_145_3 := arithEq_of_rows h (row := 145) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_3
  simp only [← c5542, ← c5543, ← c5544] at e_145_3
  have hr := e_145_3
  simp only [e_145_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1042 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 23) = bselect (a (.wire 128 55)) (a (.wire 144 15)) (a (.wire 133 27)) := by
  have c5545 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.1
  have c5546 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5547 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5548 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5549 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5550 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_4 := arithEq_of_rows h (row := 145) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_4
  simp only [← c5545, ← c5546, ← c5547] at e_145_4
  have e_145_5 := arithEq_of_rows h (row := 145) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_5
  simp only [← c5548, ← c5549, ← c5550] at e_145_5
  have hr := e_145_5
  simp only [e_145_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1043 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 146 1), a (.wire 146 2), a (.wire 146 3), a (.wire 146 4), a (.wire 146 5)] := by
  have c5551 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5552 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5553 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5554 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5555 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5556 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5557 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5558 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5559 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5560 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5561 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5562 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5563 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5564 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5565 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5566 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5567 := (leafCircuit_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5568 := (leafCircuit_copies174 a h).1
  have c5569 := (leafCircuit_copies174 a h).2.1
  have c5570 := (leafCircuit_copies174 a h).2.2.1
  have c5571 := (leafCircuit_copies174 a h).2.2.2.1
  have c5572 := (leafCircuit_copies174 a h).2.2.2.2.1
  have c5573 := (leafCircuit_copies174 a h).2.2.2.2.2.1
  have c5574 := (leafCircuit_copies174 a h).2.2.2.2.2.2.1
  have c5575 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.1
  have c5576 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.1
  have c5577 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.1
  have c5578 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5579 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5580 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5581 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5582 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5583 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5584 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5585 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5586 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5587 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5588 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5589 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5590 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5591 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5592 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5593 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5594 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5595 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5596 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5597 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5598 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5599 := (leafCircuit_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5600 := (leafCircuit_copies175 a h).1
  have c5601 := (leafCircuit_copies175 a h).2.1
  have c5602 := (leafCircuit_copies175 a h).2.2.1
  have c5603 := (leafCircuit_copies175 a h).2.2.2.1
  have c5604 := (leafCircuit_copies175 a h).2.2.2.2.1
  have c5605 := (leafCircuit_copies175 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 146) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c5551.trans k0
    · exact c5552.trans k0
    · exact c5553.trans k0
    · exact c5554.trans k0
    · exact c5555.trans k0
    · exact c5556.trans k0
    · exact c5557.trans k0
    · exact c5558.trans k0
    · exact c5559.trans k0
    · exact c5560.trans k0
    · exact c5561.trans k0
    · exact c5562.trans k0
    · exact c5563.trans k0
    · exact c5564.trans k0
    · exact c5565.trans k0
    · exact c5566.trans k0
    · exact c5567.trans k0
    · exact c5568.trans k0
    · exact c5569.trans k0
    · exact c5570.trans k0
    · exact c5571.trans k0
    · exact c5572.trans k0
    · exact c5573.trans k0
    · exact c5574.trans k0
    · exact c5575.trans k0
    · exact c5576.trans k0
    · exact c5577.trans k0
    · exact c5578.trans k0
    · exact c5579.trans k0
    · exact c5580.trans k0
    · exact c5581.trans k0
    · exact c5582.trans k0
    · exact c5583.trans k0
    · exact c5584.trans k0
    · exact c5585.trans k0
    · exact c5586.trans k0
    · exact c5587.trans k0
    · exact c5588.trans k0
    · exact c5589.trans k0
    · exact c5590.trans k0
    · exact c5591.trans k0
    · exact c5592.trans k0
    · exact c5593.trans k0
    · exact c5594.trans k0
    · exact c5595.trans k0
    · exact c5596.trans k0
    · exact c5597.trans k0
    · exact c5598.trans k0
    · exact c5599.trans k0
    · exact c5600.trans k0
    · exact c5601.trans k0
    · exact c5602.trans k0
    · exact c5603.trans k0
    · exact c5604.trans k0
    )
  rwa [c5605] at hr

theorem leafCircuit_f1044 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1045 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 5) = band (a (.virt 292)) (a (.wire 146 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1046 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 5) = band (a (.wire 146 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1047 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 5) = bor (a (.virt 14)) (a (.wire 146 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1048 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 146 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1049 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1050 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 5) = a (.virt 14) + a (.wire 146 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1051 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 5) = a (.wire 146 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1052 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 27) = bnot (a (.wire 146 5)) := by
  have c5606 := (leafCircuit_copies175 a h).2.2.2.2.2.2.1
  have c5607 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.1
  have c5608 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_6 := arithEq_of_rows h (row := 145) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_6
  simp only [← c5606, k3, ← c5607, k3, ← c5608] at e_145_6
  have hr := e_145_6
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1053 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 27) = band (a (.virt 292)) (a (.wire 145 27)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1054 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1055 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 146 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1056 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 145 27)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1057 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 5) = bor (a (.wire 146 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1058 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 4) = a (.virt 292) * a (.wire 146 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1059 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 23) = a (.virt 294) * a (.wire 146 4) := by
  have c5609 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.1
  have c5610 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5611 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_137_5 := arithEq_of_rows h (row := 137) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_5
  simp only [← c5609, k5, ← c5610, ← c5611, k5] at e_137_5
  have hr := e_137_5
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1060 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 39) = a (.virt 292) + a (.wire 146 4) := by
  have c5612 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5613 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5614 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_142_9 := arithEq_of_rows h (row := 142) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_9
  simp only [← c5612, k3, ← c5613, k3, ← c5614] at e_142_9
  have hr := e_142_9
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1061 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 31) = a (.wire 142 39) - a (.wire 137 23) := by
  have c5615 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5616 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5617 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_7 := arithEq_of_rows h (row := 145) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_7
  simp only [← c5615, ← c5616, k3, ← c5617] at e_145_7
  have hr := e_145_7
  linear_combination hr

theorem leafCircuit_f1062 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 35) = bnot (a (.wire 145 31)) := by
  have c5618 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5619 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5620 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_8 := arithEq_of_rows h (row := 145) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_8
  simp only [← c5618, k3, ← c5619, k3, ← c5620] at e_145_8
  have hr := e_145_8
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1063 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 27) = band (a (.wire 145 27)) (a (.wire 145 35)) := by
  have c5621 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5622 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5623 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_6 := arithEq_of_rows h (row := 137) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_6
  simp only [← c5621, ← c5622, ← c5623] at e_137_6
  have hr := e_137_6
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1064 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1065 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 3) = band (a (.virt 292)) (a (.wire 146 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1066 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 31) = band (a (.wire 146 3)) (a (.wire 137 27)) := by
  have c5624 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5625 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5626 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_7 := arithEq_of_rows h (row := 137) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_7
  simp only [← c5624, ← c5625, ← c5626] at e_137_7
  have hr := e_137_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1067 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 43) = bor (a (.wire 146 5)) (a (.wire 137 31)) := by
  have c5627 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5628 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5629 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5630 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5631 := (leafCircuit_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5632 := (leafCircuit_copies176 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_11 := arithEq_of_rows h (row := 104) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_11
  simp only [← c5627, ← c5628, ← c5629] at e_104_11
  have e_142_10 := arithEq_of_rows h (row := 142) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_10
  simp only [← c5630, ← c5631, k3, ← c5632] at e_142_10
  have hr := e_142_10
  simp only [e_104_11] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1068 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 146 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1069 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1070 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 3) = a (.virt 14) + a (.wire 146 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1071 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 3) = a (.wire 146 3) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1072 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 39) = bnot (a (.wire 146 3)) := by
  have c5633 := (leafCircuit_copies176 a h).2.1
  have c5634 := (leafCircuit_copies176 a h).2.2.1
  have c5635 := (leafCircuit_copies176 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_9 := arithEq_of_rows h (row := 145) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_9
  simp only [← c5633, k3, ← c5634, k3, ← c5635] at e_145_9
  have hr := e_145_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1073 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 35) = band (a (.wire 137 27)) (a (.wire 145 39)) := by
  have c5636 := (leafCircuit_copies176 a h).2.2.2.2.1
  have c5637 := (leafCircuit_copies176 a h).2.2.2.2.2.1
  have c5638 := (leafCircuit_copies176 a h).2.2.2.2.2.2.1
  have e_137_8 := arithEq_of_rows h (row := 137) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_8
  simp only [← c5636, ← c5637, ← c5638] at e_137_8
  have hr := e_137_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1074 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1075 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 146 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1076 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 137 35)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1077 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 43) = bor (a (.wire 142 43)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1078 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 2) = a (.virt 292) * a (.wire 146 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1079 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 39) = a (.virt 294) * a (.wire 146 2) := by
  have c5639 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.1
  have c5640 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.1
  have c5641 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_137_9 := arithEq_of_rows h (row := 137) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_9
  simp only [← c5639, k5, ← c5640, ← c5641, k5] at e_137_9
  have hr := e_137_9
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1080 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 47) = a (.virt 292) + a (.wire 146 2) := by
  have c5642 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5643 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5644 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_142_11 := arithEq_of_rows h (row := 142) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_11
  simp only [← c5642, k3, ← c5643, k3, ← c5644] at e_142_11
  have hr := e_142_11
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1081 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 43) = a (.wire 142 47) - a (.wire 137 39) := by
  have c5645 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5646 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5647 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_10 := arithEq_of_rows h (row := 145) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_10
  simp only [← c5645, ← c5646, k3, ← c5647] at e_145_10
  have hr := e_145_10
  linear_combination hr

theorem leafCircuit_f1082 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 47) = bnot (a (.wire 145 43)) := by
  have c5648 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5649 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5650 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_11 := arithEq_of_rows h (row := 145) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_11
  simp only [← c5648, k3, ← c5649, k3, ← c5650] at e_145_11
  have hr := e_145_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1083 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 43) = band (a (.wire 137 35)) (a (.wire 145 47)) := by
  have c5651 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5652 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5653 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_10 := arithEq_of_rows h (row := 137) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_10
  simp only [← c5651, ← c5652, ← c5653] at e_137_10
  have hr := e_137_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1084 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1085 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 146 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1086 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 137 43)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1087 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 43) = bor (a (.wire 142 43)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1088 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 146 1) = a (.virt 292) * a (.wire 146 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1089 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 47) = a (.virt 294) * a (.wire 146 1) := by
  have c5654 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5655 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5656 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_137_11 := arithEq_of_rows h (row := 137) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_11
  simp only [← c5654, k5, ← c5655, ← c5656, k5] at e_137_11
  have hr := e_137_11
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1090 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 51) = a (.virt 292) + a (.wire 146 1) := by
  have c5657 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5658 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5659 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_142_12 := arithEq_of_rows h (row := 142) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_12
  simp only [← c5657, k3, ← c5658, k3, ← c5659] at e_142_12
  have hr := e_142_12
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1091 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 51) = a (.wire 142 51) - a (.wire 137 47) := by
  have c5660 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5661 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5662 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_12 := arithEq_of_rows h (row := 145) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_12
  simp only [← c5660, ← c5661, k3, ← c5662] at e_145_12
  have hr := e_145_12
  linear_combination hr

theorem leafCircuit_f1092 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 145 55) = bnot (a (.wire 145 51)) := by
  have c5663 := (leafCircuit_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5664 := (leafCircuit_copies177 a h).1
  have c5665 := (leafCircuit_copies177 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_145_13 := arithEq_of_rows h (row := 145) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_13
  simp only [← c5663, k3, ← c5664, k3, ← c5665] at e_145_13
  have hr := e_145_13
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1093 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 137 51) = band (a (.wire 137 43)) (a (.wire 145 55)) := by
  have c5666 := (leafCircuit_copies177 a h).2.2.1
  have c5667 := (leafCircuit_copies177 a h).2.2.2.1
  have c5668 := (leafCircuit_copies177 a h).2.2.2.2.1
  have e_137_12 := arithEq_of_rows h (row := 137) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_12
  simp only [← c5666, ← c5667, ← c5668] at e_137_12
  have hr := e_137_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1094 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 218)) 2 := by
  have c5669 := (leafCircuit_copies177 a h).2.2.2.2.2.1
  have c5670 := (leafCircuit_copies177 a h).2.2.2.2.2.2.1
  have c5671 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.1
  have c5672 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.1
  have c5673 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.1
  have c5674 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5675 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5676 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5677 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5678 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5679 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5680 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5681 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5682 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5683 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5684 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5685 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5686 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5687 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5688 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5689 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5690 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5691 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5692 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5693 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5694 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5695 := (leafCircuit_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5696 := (leafCircuit_copies178 a h).1
  have c5697 := (leafCircuit_copies178 a h).2.1
  have c5698 := (leafCircuit_copies178 a h).2.2.1
  have c5699 := (leafCircuit_copies178 a h).2.2.2.1
  have c5700 := (leafCircuit_copies178 a h).2.2.2.2.1
  have c5701 := (leafCircuit_copies178 a h).2.2.2.2.2.1
  have c5702 := (leafCircuit_copies178 a h).2.2.2.2.2.2.1
  have c5703 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.1
  have c5704 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.1
  have c5705 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.1
  have c5706 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5707 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5708 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5709 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5710 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5711 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5712 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5713 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5714 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5715 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5716 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5717 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5718 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5719 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5720 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5721 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5722 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5723 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5724 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5725 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5726 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 147) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c5669.trans k0
    · exact c5670.trans k0
    · exact c5671.trans k0
    · exact c5672.trans k0
    · exact c5673.trans k0
    · exact c5674.trans k0
    · exact c5675.trans k0
    · exact c5676.trans k0
    · exact c5677.trans k0
    · exact c5678.trans k0
    · exact c5679.trans k0
    · exact c5680.trans k0
    · exact c5681.trans k0
    · exact c5682.trans k0
    · exact c5683.trans k0
    · exact c5684.trans k0
    · exact c5685.trans k0
    · exact c5686.trans k0
    · exact c5687.trans k0
    · exact c5688.trans k0
    · exact c5689.trans k0
    · exact c5690.trans k0
    · exact c5691.trans k0
    · exact c5692.trans k0
    · exact c5693.trans k0
    · exact c5694.trans k0
    · exact c5695.trans k0
    · exact c5696.trans k0
    · exact c5697.trans k0
    · exact c5698.trans k0
    · exact c5699.trans k0
    · exact c5700.trans k0
    · exact c5701.trans k0
    · exact c5702.trans k0
    · exact c5703.trans k0
    · exact c5704.trans k0
    · exact c5705.trans k0
    · exact c5706.trans k0
    · exact c5707.trans k0
    · exact c5708.trans k0
    · exact c5709.trans k0
    · exact c5710.trans k0
    · exact c5711.trans k0
    · exact c5712.trans k0
    · exact c5713.trans k0
    · exact c5714.trans k0
    · exact c5715.trans k0
    · exact c5716.trans k0
    · exact c5717.trans k0
    · exact c5718.trans k0
    · exact c5719.trans k0
    · exact c5720.trans k0
    · exact c5721.trans k0
    · exact c5722.trans k0
    · exact c5723.trans k0
    · exact c5724.trans k0
    · exact c5725.trans k0
    )
  rwa [c5726] at hr

theorem leafCircuit_f1095 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 218)) (a (.virt 14)) (a (.virt 384)) (a (.virt 385)) := by
  have c5727 := (leafCircuit_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5728 := (leafCircuit_copies179 a h).1
  have c5729 := (leafCircuit_copies179 a h).2.1
  have c5730 := (leafCircuit_copies179 a h).2.2.1
  have c5731 := (leafCircuit_copies179 a h).2.2.2.1
  have c5732 := (leafCircuit_copies179 a h).2.2.2.2.1
  have c5733 := (leafCircuit_copies179 a h).2.2.2.2.2.1
  have c5734 := (leafCircuit_copies179 a h).2.2.2.2.2.2.1
  have c5735 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.1
  have c5736 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.1
  have c5737 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.1
  have c5738 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5739 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5740 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_137_13 := arithEq_of_rows h (row := 137) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_13
  simp only [← c5730, ← c5731, ← c5732] at e_137_13
  have e_137_14 := arithEq_of_rows h (row := 137) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_14
  simp only [← c5733, ← c5734, ← c5735] at e_137_14
  have e_145_14 := arithEq_of_rows h (row := 145) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_14
  simp only [← c5727, k3, ← c5728, k3, ← c5729] at e_145_14
  have e_148_0 := arithEq_of_rows h (row := 148) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_0
  simp only [← c5736, ← c5737, k3, ← c5738] at e_148_0
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_137_13
    linear_combination c5739.trans k0 - hc
  · have hc := e_148_0
    simp only [e_145_14, e_137_14] at hc
    linear_combination c5740.trans k0 - hc

theorem leafCircuit_f1096 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 218)) (a (.virt 292)) (a (.virt 386)) (a (.virt 387)) := by
  have c5741 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5742 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5743 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5744 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5745 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5746 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5747 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5748 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5749 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5750 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5751 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5752 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5753 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5754 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5755 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5756 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5757 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_148_1 := arithEq_of_rows h (row := 148) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_1
  simp only [← c5741, k3, ← c5742, k3, ← c5743] at e_148_1
  have e_148_2 := arithEq_of_rows h (row := 148) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_2
  simp only [← c5744, ← c5745, k3, ← c5746, k3] at e_148_2
  have e_148_3 := arithEq_of_rows h (row := 148) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_3
  simp only [← c5753, ← c5754, k3, ← c5755] at e_148_3
  have e_149_0 := arithEq_of_rows h (row := 149) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_0
  simp only [← c5747, ← c5748, ← c5749] at e_149_0
  have e_149_1 := arithEq_of_rows h (row := 149) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_1
  simp only [← c5750, ← c5751, ← c5752] at e_149_1
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_149_0
    simp only [e_148_2] at hc
    linear_combination c5756.trans k0 - hc
  · have hc := e_148_3
    simp only [e_148_1, e_149_1, e_148_2] at hc
    linear_combination c5757.trans k0 - hc

theorem leafCircuit_f1097 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 218)) (a (.virt 294)) (a (.virt 388)) (a (.virt 389)) := by
  have c5758 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5759 := (leafCircuit_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5760 := (leafCircuit_copies180 a h).1
  have c5761 := (leafCircuit_copies180 a h).2.1
  have c5762 := (leafCircuit_copies180 a h).2.2.1
  have c5763 := (leafCircuit_copies180 a h).2.2.2.1
  have c5764 := (leafCircuit_copies180 a h).2.2.2.2.1
  have c5765 := (leafCircuit_copies180 a h).2.2.2.2.2.1
  have c5766 := (leafCircuit_copies180 a h).2.2.2.2.2.2.1
  have c5767 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.1
  have c5768 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.1
  have c5769 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.1
  have c5770 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5771 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5772 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5773 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5774 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_148_4 := arithEq_of_rows h (row := 148) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_4
  simp only [← c5758, k3, ← c5759, k3, ← c5760] at e_148_4
  have e_148_5 := arithEq_of_rows h (row := 148) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_5
  simp only [← c5761, ← c5762, k3, ← c5763, k5] at e_148_5
  have e_148_6 := arithEq_of_rows h (row := 148) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_6
  simp only [← c5770, ← c5771, k3, ← c5772] at e_148_6
  have e_149_2 := arithEq_of_rows h (row := 149) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_2
  simp only [← c5764, ← c5765, ← c5766] at e_149_2
  have e_149_3 := arithEq_of_rows h (row := 149) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_3
  simp only [← c5767, ← c5768, ← c5769] at e_149_3
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_149_2
    simp only [e_148_5] at hc
    linear_combination c5773.trans k0 - hc
  · have hc := e_148_6
    simp only [e_148_4, e_149_3, e_148_5] at hc
    linear_combination c5774.trans k0 - hc

theorem leafCircuit_f1098 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 218)) (a (.virt 295)) (a (.virt 390)) (a (.virt 391)) := by
  have c5775 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5776 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5777 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5778 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5779 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5780 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5781 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5782 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5783 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5784 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5785 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5786 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5787 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5788 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5789 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5790 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5791 := (leafCircuit_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_148_7 := arithEq_of_rows h (row := 148) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_7
  simp only [← c5775, k3, ← c5776, k3, ← c5777] at e_148_7
  have e_148_8 := arithEq_of_rows h (row := 148) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_8
  simp only [← c5778, ← c5779, k3, ← c5780, k6] at e_148_8
  have e_148_9 := arithEq_of_rows h (row := 148) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_9
  simp only [← c5787, ← c5788, k3, ← c5789] at e_148_9
  have e_149_4 := arithEq_of_rows h (row := 149) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_4
  simp only [← c5781, ← c5782, ← c5783] at e_149_4
  have e_149_5 := arithEq_of_rows h (row := 149) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_5
  simp only [← c5784, ← c5785, ← c5786] at e_149_5
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_149_4
    simp only [e_148_8] at hc
    linear_combination c5790.trans k0 - hc
  · have hc := e_148_9
    simp only [e_148_7, e_149_5, e_148_8] at hc
    linear_combination c5791.trans k0 - hc

theorem leafCircuit_f1099 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 148 47) = bselect (a (.virt 384)) (a (.wire 140 59)) (a (.virt 147)) := by
  have c5792 := (leafCircuit_copies181 a h).1
  have c5793 := (leafCircuit_copies181 a h).2.1
  have c5794 := (leafCircuit_copies181 a h).2.2.1
  have c5795 := (leafCircuit_copies181 a h).2.2.2.1
  have c5796 := (leafCircuit_copies181 a h).2.2.2.2.1
  have c5797 := (leafCircuit_copies181 a h).2.2.2.2.2.1
  have e_148_10 := arithEq_of_rows h (row := 148) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_10
  simp only [← c5792, ← c5793, ← c5794] at e_148_10
  have e_148_11 := arithEq_of_rows h (row := 148) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_11
  simp only [← c5795, ← c5796, ← c5797] at e_148_11
  have hr := e_148_11
  simp only [e_148_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1100 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 148 55) = bselect (a (.virt 384)) (a (.wire 145 7)) (a (.virt 148)) := by
  have c5798 := (leafCircuit_copies181 a h).2.2.2.2.2.2.1
  have c5799 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.1
  have c5800 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.1
  have c5801 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.1
  have c5802 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5803 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_12 := arithEq_of_rows h (row := 148) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_12
  simp only [← c5798, ← c5799, ← c5800] at e_148_12
  have e_148_13 := arithEq_of_rows h (row := 148) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_13
  simp only [← c5801, ← c5802, ← c5803] at e_148_13
  have hr := e_148_13
  simp only [e_148_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1101 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 3) = bselect (a (.virt 384)) (a (.wire 145 15)) (a (.virt 149)) := by
  have c5804 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5805 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5806 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5807 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5808 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5809 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_14 := arithEq_of_rows h (row := 148) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_14
  simp only [← c5804, ← c5805, ← c5806] at e_148_14
  have e_150_0 := arithEq_of_rows h (row := 150) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_0
  simp only [← c5807, ← c5808, ← c5809] at e_150_0
  have hr := e_150_0
  simp only [e_148_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1102 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 11) = bselect (a (.virt 384)) (a (.wire 145 23)) (a (.virt 150)) := by
  have c5810 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5811 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5812 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5813 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5814 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5815 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_1 := arithEq_of_rows h (row := 150) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_1
  simp only [← c5810, ← c5811, ← c5812] at e_150_1
  have e_150_2 := arithEq_of_rows h (row := 150) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_2
  simp only [← c5813, ← c5814, ← c5815] at e_150_2
  have hr := e_150_2
  simp only [e_150_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1103 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 19) = bselect (a (.virt 384)) (a (.virt 147)) (a (.virt 151)) := by
  have c5816 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5817 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5818 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5819 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5820 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5821 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_3 := arithEq_of_rows h (row := 150) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_3
  simp only [← c5816, ← c5817, ← c5818] at e_150_3
  have e_150_4 := arithEq_of_rows h (row := 150) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_4
  simp only [← c5819, ← c5820, ← c5821] at e_150_4
  have hr := e_150_4
  simp only [e_150_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1104 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 27) = bselect (a (.virt 386)) (a (.wire 140 59)) (a (.wire 150 19)) := by
  have c5822 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5823 := (leafCircuit_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5824 := (leafCircuit_copies182 a h).1
  have c5825 := (leafCircuit_copies182 a h).2.1
  have c5826 := (leafCircuit_copies182 a h).2.2.1
  have c5827 := (leafCircuit_copies182 a h).2.2.2.1
  have e_150_5 := arithEq_of_rows h (row := 150) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_5
  simp only [← c5822, ← c5823, ← c5824] at e_150_5
  have e_150_6 := arithEq_of_rows h (row := 150) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_6
  simp only [← c5825, ← c5826, ← c5827] at e_150_6
  have hr := e_150_6
  simp only [e_150_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1105 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 35) = bselect (a (.virt 384)) (a (.virt 148)) (a (.virt 152)) := by
  have c5828 := (leafCircuit_copies182 a h).2.2.2.2.1
  have c5829 := (leafCircuit_copies182 a h).2.2.2.2.2.1
  have c5830 := (leafCircuit_copies182 a h).2.2.2.2.2.2.1
  have c5831 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.1
  have c5832 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.1
  have c5833 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.1
  have e_150_7 := arithEq_of_rows h (row := 150) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_7
  simp only [← c5828, ← c5829, ← c5830] at e_150_7
  have e_150_8 := arithEq_of_rows h (row := 150) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_8
  simp only [← c5831, ← c5832, ← c5833] at e_150_8
  have hr := e_150_8
  simp only [e_150_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1106 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 43) = bselect (a (.virt 386)) (a (.wire 145 7)) (a (.wire 150 35)) := by
  have c5834 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5835 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5836 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5837 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5838 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5839 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_9 := arithEq_of_rows h (row := 150) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_9
  simp only [← c5834, ← c5835, ← c5836] at e_150_9
  have e_150_10 := arithEq_of_rows h (row := 150) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_10
  simp only [← c5837, ← c5838, ← c5839] at e_150_10
  have hr := e_150_10
  simp only [e_150_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1107 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 51) = bselect (a (.virt 384)) (a (.virt 149)) (a (.virt 153)) := by
  have c5840 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5841 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5842 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5843 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5844 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5845 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_11 := arithEq_of_rows h (row := 150) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_11
  simp only [← c5840, ← c5841, ← c5842] at e_150_11
  have e_150_12 := arithEq_of_rows h (row := 150) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_12
  simp only [← c5843, ← c5844, ← c5845] at e_150_12
  have hr := e_150_12
  simp only [e_150_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1108 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 150 59) = bselect (a (.virt 386)) (a (.wire 145 15)) (a (.wire 150 51)) := by
  have c5846 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5847 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5848 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5849 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5850 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5851 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_13 := arithEq_of_rows h (row := 150) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_13
  simp only [← c5846, ← c5847, ← c5848] at e_150_13
  have e_150_14 := arithEq_of_rows h (row := 150) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_14
  simp only [← c5849, ← c5850, ← c5851] at e_150_14
  have hr := e_150_14
  simp only [e_150_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1109 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 7) = bselect (a (.virt 384)) (a (.virt 150)) (a (.virt 154)) := by
  have c5852 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5853 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5854 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5855 := (leafCircuit_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5856 := (leafCircuit_copies183 a h).1
  have c5857 := (leafCircuit_copies183 a h).2.1
  have e_151_0 := arithEq_of_rows h (row := 151) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_0
  simp only [← c5852, ← c5853, ← c5854] at e_151_0
  have e_151_1 := arithEq_of_rows h (row := 151) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_1
  simp only [← c5855, ← c5856, ← c5857] at e_151_1
  have hr := e_151_1
  simp only [e_151_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1110 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 15) = bselect (a (.virt 386)) (a (.wire 145 23)) (a (.wire 151 7)) := by
  have c5858 := (leafCircuit_copies183 a h).2.2.1
  have c5859 := (leafCircuit_copies183 a h).2.2.2.1
  have c5860 := (leafCircuit_copies183 a h).2.2.2.2.1
  have c5861 := (leafCircuit_copies183 a h).2.2.2.2.2.1
  have c5862 := (leafCircuit_copies183 a h).2.2.2.2.2.2.1
  have c5863 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.1
  have e_151_2 := arithEq_of_rows h (row := 151) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_2
  simp only [← c5858, ← c5859, ← c5860] at e_151_2
  have e_151_3 := arithEq_of_rows h (row := 151) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_3
  simp only [← c5861, ← c5862, ← c5863] at e_151_3
  have hr := e_151_3
  simp only [e_151_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1111 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) := by
  have c5864 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.1
  have c5865 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.1
  have c5866 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5867 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5868 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5869 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c5864, ← c5865, ← c5866] at e_104_12
  have e_142_13 := arithEq_of_rows h (row := 142) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_13
  simp only [← c5867, ← c5868, k3, ← c5869] at e_142_13
  have hr := e_142_13
  simp only [e_104_12] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1112 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 23) = bselect (a (.wire 142 55)) (a (.virt 151)) (a (.virt 155)) := by
  have c5870 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5871 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5872 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5873 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5874 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5875 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_4 := arithEq_of_rows h (row := 151) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_4
  simp only [← c5870, ← c5871, ← c5872] at e_151_4
  have e_151_5 := arithEq_of_rows h (row := 151) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_5
  simp only [← c5873, ← c5874, ← c5875] at e_151_5
  have hr := e_151_5
  simp only [e_151_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1113 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 31) = bselect (a (.virt 388)) (a (.wire 140 59)) (a (.wire 151 23)) := by
  have c5876 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5877 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5878 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5879 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5880 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5881 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_6 := arithEq_of_rows h (row := 151) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_6
  simp only [← c5876, ← c5877, ← c5878] at e_151_6
  have e_151_7 := arithEq_of_rows h (row := 151) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_7
  simp only [← c5879, ← c5880, ← c5881] at e_151_7
  have hr := e_151_7
  simp only [e_151_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1114 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) := by
  have c5864 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.1
  have c5865 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.1
  have c5866 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5867 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5868 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5869 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c5864, ← c5865, ← c5866] at e_104_12
  have e_142_13 := arithEq_of_rows h (row := 142) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_13
  simp only [← c5867, ← c5868, k3, ← c5869] at e_142_13
  have hr := e_142_13
  simp only [e_104_12] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1115 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 39) = bselect (a (.wire 142 55)) (a (.virt 152)) (a (.virt 156)) := by
  have c5882 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5883 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5884 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5885 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5886 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5887 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_151_8 := arithEq_of_rows h (row := 151) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_8
  simp only [← c5882, ← c5883, ← c5884] at e_151_8
  have e_151_9 := arithEq_of_rows h (row := 151) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_9
  simp only [← c5885, ← c5886, ← c5887] at e_151_9
  have hr := e_151_9
  simp only [e_151_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1116 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 47) = bselect (a (.virt 388)) (a (.wire 145 7)) (a (.wire 151 39)) := by
  have c5888 := (leafCircuit_copies184 a h).1
  have c5889 := (leafCircuit_copies184 a h).2.1
  have c5890 := (leafCircuit_copies184 a h).2.2.1
  have c5891 := (leafCircuit_copies184 a h).2.2.2.1
  have c5892 := (leafCircuit_copies184 a h).2.2.2.2.1
  have c5893 := (leafCircuit_copies184 a h).2.2.2.2.2.1
  have e_151_10 := arithEq_of_rows h (row := 151) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_10
  simp only [← c5888, ← c5889, ← c5890] at e_151_10
  have e_151_11 := arithEq_of_rows h (row := 151) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_11
  simp only [← c5891, ← c5892, ← c5893] at e_151_11
  have hr := e_151_11
  simp only [e_151_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1117 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) := by
  have c5864 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.1
  have c5865 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.1
  have c5866 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5867 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5868 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5869 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c5864, ← c5865, ← c5866] at e_104_12
  have e_142_13 := arithEq_of_rows h (row := 142) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_13
  simp only [← c5867, ← c5868, k3, ← c5869] at e_142_13
  have hr := e_142_13
  simp only [e_104_12] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1118 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 151 55) = bselect (a (.wire 142 55)) (a (.virt 153)) (a (.virt 157)) := by
  have c5894 := (leafCircuit_copies184 a h).2.2.2.2.2.2.1
  have c5895 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.1
  have c5896 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.1
  have c5897 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.1
  have c5898 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5899 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_12 := arithEq_of_rows h (row := 151) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_12
  simp only [← c5894, ← c5895, ← c5896] at e_151_12
  have e_151_13 := arithEq_of_rows h (row := 151) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_13
  simp only [← c5897, ← c5898, ← c5899] at e_151_13
  have hr := e_151_13
  simp only [e_151_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1119 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 3) = bselect (a (.virt 388)) (a (.wire 145 15)) (a (.wire 151 55)) := by
  have c5900 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5901 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5902 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5903 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5904 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5905 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_14 := arithEq_of_rows h (row := 151) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_14
  simp only [← c5900, ← c5901, ← c5902] at e_151_14
  have e_152_0 := arithEq_of_rows h (row := 152) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_0
  simp only [← c5903, ← c5904, ← c5905] at e_152_0
  have hr := e_152_0
  simp only [e_151_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1120 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) := by
  have c5864 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.1
  have c5865 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.1
  have c5866 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5867 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5868 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5869 := (leafCircuit_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c5864, ← c5865, ← c5866] at e_104_12
  have e_142_13 := arithEq_of_rows h (row := 142) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_13
  simp only [← c5867, ← c5868, k3, ← c5869] at e_142_13
  have hr := e_142_13
  simp only [e_104_12] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1121 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 11) = bselect (a (.wire 142 55)) (a (.virt 154)) (a (.virt 158)) := by
  have c5906 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5907 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5908 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5909 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5910 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5911 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_1 := arithEq_of_rows h (row := 152) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_1
  simp only [← c5906, ← c5907, ← c5908] at e_152_1
  have e_152_2 := arithEq_of_rows h (row := 152) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_2
  simp only [← c5909, ← c5910, ← c5911] at e_152_2
  have hr := e_152_2
  simp only [e_152_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1122 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 19) = bselect (a (.virt 388)) (a (.wire 145 23)) (a (.wire 152 11)) := by
  have c5912 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5913 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5914 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5915 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5916 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5917 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_3 := arithEq_of_rows h (row := 152) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_3
  simp only [← c5912, ← c5913, ← c5914] at e_152_3
  have e_152_4 := arithEq_of_rows h (row := 152) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_4
  simp only [← c5915, ← c5916, ← c5917] at e_152_4
  have hr := e_152_4
  simp only [e_152_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1123 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 27) = bselect (a (.virt 390)) (a (.wire 140 59)) (a (.virt 155)) := by
  have c5918 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5919 := (leafCircuit_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5920 := (leafCircuit_copies185 a h).1
  have c5921 := (leafCircuit_copies185 a h).2.1
  have c5922 := (leafCircuit_copies185 a h).2.2.1
  have c5923 := (leafCircuit_copies185 a h).2.2.2.1
  have e_152_5 := arithEq_of_rows h (row := 152) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_5
  simp only [← c5918, ← c5919, ← c5920] at e_152_5
  have e_152_6 := arithEq_of_rows h (row := 152) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_6
  simp only [← c5921, ← c5922, ← c5923] at e_152_6
  have hr := e_152_6
  simp only [e_152_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1124 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 35) = bselect (a (.virt 390)) (a (.wire 145 7)) (a (.virt 156)) := by
  have c5924 := (leafCircuit_copies185 a h).2.2.2.2.1
  have c5925 := (leafCircuit_copies185 a h).2.2.2.2.2.1
  have c5926 := (leafCircuit_copies185 a h).2.2.2.2.2.2.1
  have c5927 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.1
  have c5928 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.1
  have c5929 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.1
  have e_152_7 := arithEq_of_rows h (row := 152) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_7
  simp only [← c5924, ← c5925, ← c5926] at e_152_7
  have e_152_8 := arithEq_of_rows h (row := 152) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_8
  simp only [← c5927, ← c5928, ← c5929] at e_152_8
  have hr := e_152_8
  simp only [e_152_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1125 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 43) = bselect (a (.virt 390)) (a (.wire 145 15)) (a (.virt 157)) := by
  have c5930 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5931 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5932 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5933 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5934 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5935 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_9 := arithEq_of_rows h (row := 152) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_9
  simp only [← c5930, ← c5931, ← c5932] at e_152_9
  have e_152_10 := arithEq_of_rows h (row := 152) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_10
  simp only [← c5933, ← c5934, ← c5935] at e_152_10
  have hr := e_152_10
  simp only [e_152_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1126 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 51) = bselect (a (.virt 390)) (a (.wire 145 23)) (a (.virt 158)) := by
  have c5936 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5937 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5938 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5939 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5940 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5941 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_11 := arithEq_of_rows h (row := 152) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_11
  simp only [← c5936, ← c5937, ← c5938] at e_152_11
  have e_152_12 := arithEq_of_rows h (row := 152) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_12
  simp only [← c5939, ← c5940, ← c5941] at e_152_12
  have hr := e_152_12
  simp only [e_152_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1127 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 156 12) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 0 ∧ a (.wire 156 13) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 1 ∧ a (.wire 156 14) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 2 ∧ a (.wire 156 15) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 3) := by
  have c5942 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5943 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5944 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5945 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5946 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5947 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5948 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5949 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5950 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5951 := (leafCircuit_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5952 := (leafCircuit_copies186 a h).1
  have c5953 := (leafCircuit_copies186 a h).2.1
  have c5954 := (leafCircuit_copies186 a h).2.2.1
  have c5955 := (leafCircuit_copies186 a h).2.2.2.1
  have c5956 := (leafCircuit_copies186 a h).2.2.2.2.1
  have c5957 := (leafCircuit_copies186 a h).2.2.2.2.2.1
  have c5958 := (leafCircuit_copies186 a h).2.2.2.2.2.2.1
  have c5959 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.1
  have c5960 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.1
  have c5961 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.1
  have c5962 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5963 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5964 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5965 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5966 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5967 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5968 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5969 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5970 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5971 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5972 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5973 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5974 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5975 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5976 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5977 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5978 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5979 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5980 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5981 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5982 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5983 := (leafCircuit_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5984 := (leafCircuit_copies187 a h).1
  have c5985 := (leafCircuit_copies187 a h).2.1
  have c5986 := (leafCircuit_copies187 a h).2.2.1
  have c5987 := (leafCircuit_copies187 a h).2.2.2.1
  have c5988 := (leafCircuit_copies187 a h).2.2.2.2.1
  have c5989 := (leafCircuit_copies187 a h).2.2.2.2.2.1
  have c5990 := (leafCircuit_copies187 a h).2.2.2.2.2.2.1
  have c5991 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.1
  have c5992 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.1
  have c5993 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.1
  have c5994 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5995 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5996 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5997 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5998 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5999 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6000 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6001 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6002 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6003 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6004 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_142_14 := arithEq_of_rows h (row := 142) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_14
  simp only [← c5954, ← c5955, k3, ← c5956] at e_142_14
  have e_154_0 := arithEq_of_rows h (row := 154) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_0
  simp only [← c5957, ← c5958, k3, ← c5959] at e_154_0
  have e_154_1 := arithEq_of_rows h (row := 154) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_1
  simp only [← c5960, ← c5961, k3, ← c5962] at e_154_1
  have e_154_2 := arithEq_of_rows h (row := 154) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_2
  simp only [← c5963, ← c5964, k3, ← c5965] at e_154_2
  have e_154_3 := arithEq_of_rows h (row := 154) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_3
  simp only [← c5966, ← c5967, k3, ← c5968] at e_154_3
  have e_154_4 := arithEq_of_rows h (row := 154) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_4
  simp only [← c5969, ← c5970, k3, ← c5971] at e_154_4
  have e_154_5 := arithEq_of_rows h (row := 154) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_5
  simp only [← c5972, ← c5973, k3, ← c5974] at e_154_5
  have e_154_6 := arithEq_of_rows h (row := 154) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_6
  simp only [← c5975, ← c5976, k3, ← c5977] at e_154_6
  have e_154_7 := arithEq_of_rows h (row := 154) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_7
  simp only [← c5990, ← c5991, k3, ← c5992, k3] at e_154_7
  have hin0 : poseidon2In a 153 = addBlock (fun _ => 0) [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15)] :=
    poseidon2In_first c5942.symm c5943.symm c5944.symm c5945.symm c5946.symm c5947.symm c5948.symm c5949.symm (c5950.symm.trans k0) (c5951.symm.trans k0) (c5952.symm.trans k0) (c5953.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 153) rfl rfl hin0
  have hin1 : poseidon2In a 155 = addBlock (poseidon2Out a 153) [a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] :=
    poseidon2In_chain (by linear_combination -c5978 + e_142_14) (by linear_combination -c5979 + e_154_0) (by linear_combination -c5980 + e_154_1) (by linear_combination -c5981 + e_154_2) (by linear_combination -c5982 + e_154_3) (by linear_combination -c5983 + e_154_4) (by linear_combination -c5984 + e_154_5) (by linear_combination -c5985 + e_154_6) c5986.symm c5987.symm c5988.symm c5989.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 155) rfl rfl hin1
  have hin2 : poseidon2In a 156 = addBlock (poseidon2Out a 155) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c5993 + e_154_7) (c5994.symm.trans (add_zero _).symm) (c5995.symm.trans (add_zero _).symm) (c5996.symm.trans (add_zero _).symm) (c5997.symm.trans (add_zero _).symm) (c5998.symm.trans (add_zero _).symm) (c5999.symm.trans (add_zero _).symm) (c6000.symm.trans (add_zero _).symm) c6001.symm c6002.symm c6003.symm c6004.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 156) rfl rfl hin2
  have hpad : pad10 [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] =
      [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15)] ++ ([a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1128 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 152 59) = bselect (a (.wire 142 43)) (a (.wire 156 12)) (a (.wire 140 59)) := by
  have c6005 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6006 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6007 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6008 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6009 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6010 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_13 := arithEq_of_rows h (row := 152) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_13
  simp only [← c6005, ← c6006, ← c6007] at e_152_13
  have e_152_14 := arithEq_of_rows h (row := 152) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_14
  simp only [← c6008, ← c6009, ← c6010] at e_152_14
  have hr := e_152_14
  simp only [e_152_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1129 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 7) = bselect (a (.wire 142 43)) (a (.wire 156 13)) (a (.wire 145 7)) := by
  have c6011 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6012 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6013 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6014 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6015 := (leafCircuit_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6016 := (leafCircuit_copies188 a h).1
  have e_157_0 := arithEq_of_rows h (row := 157) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_0
  simp only [← c6011, ← c6012, ← c6013] at e_157_0
  have e_157_1 := arithEq_of_rows h (row := 157) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_1
  simp only [← c6014, ← c6015, ← c6016] at e_157_1
  have hr := e_157_1
  simp only [e_157_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1130 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 15) = bselect (a (.wire 142 43)) (a (.wire 156 14)) (a (.wire 145 15)) := by
  have c6017 := (leafCircuit_copies188 a h).2.1
  have c6018 := (leafCircuit_copies188 a h).2.2.1
  have c6019 := (leafCircuit_copies188 a h).2.2.2.1
  have c6020 := (leafCircuit_copies188 a h).2.2.2.2.1
  have c6021 := (leafCircuit_copies188 a h).2.2.2.2.2.1
  have c6022 := (leafCircuit_copies188 a h).2.2.2.2.2.2.1
  have e_157_2 := arithEq_of_rows h (row := 157) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_2
  simp only [← c6017, ← c6018, ← c6019] at e_157_2
  have e_157_3 := arithEq_of_rows h (row := 157) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_3
  simp only [← c6020, ← c6021, ← c6022] at e_157_3
  have hr := e_157_3
  simp only [e_157_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1131 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 23) = bselect (a (.wire 142 43)) (a (.wire 156 15)) (a (.wire 145 23)) := by
  have c6023 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.1
  have c6024 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.1
  have c6025 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.1
  have c6026 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6027 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6028 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_4 := arithEq_of_rows h (row := 157) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_4
  simp only [← c6023, ← c6024, ← c6025] at e_157_4
  have e_157_5 := arithEq_of_rows h (row := 157) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_5
  simp only [← c6026, ← c6027, ← c6028] at e_157_5
  have hr := e_157_5
  simp only [e_157_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1132 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 158 1), a (.wire 158 2), a (.wire 158 3), a (.wire 158 4), a (.wire 158 5)] := by
  have c6029 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6030 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6031 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6032 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6033 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6034 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6035 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6036 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6037 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6038 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6039 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6040 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6041 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6042 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6043 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6044 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6045 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6046 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6047 := (leafCircuit_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6048 := (leafCircuit_copies189 a h).1
  have c6049 := (leafCircuit_copies189 a h).2.1
  have c6050 := (leafCircuit_copies189 a h).2.2.1
  have c6051 := (leafCircuit_copies189 a h).2.2.2.1
  have c6052 := (leafCircuit_copies189 a h).2.2.2.2.1
  have c6053 := (leafCircuit_copies189 a h).2.2.2.2.2.1
  have c6054 := (leafCircuit_copies189 a h).2.2.2.2.2.2.1
  have c6055 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.1
  have c6056 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.1
  have c6057 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.1
  have c6058 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6059 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6060 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6061 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6062 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6063 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6064 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6065 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6066 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6067 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6068 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6069 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6070 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6071 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6072 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6073 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6074 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6075 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6076 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6077 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6078 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6079 := (leafCircuit_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6080 := (leafCircuit_copies190 a h).1
  have c6081 := (leafCircuit_copies190 a h).2.1
  have c6082 := (leafCircuit_copies190 a h).2.2.1
  have c6083 := (leafCircuit_copies190 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 158) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c6029.trans k0
    · exact c6030.trans k0
    · exact c6031.trans k0
    · exact c6032.trans k0
    · exact c6033.trans k0
    · exact c6034.trans k0
    · exact c6035.trans k0
    · exact c6036.trans k0
    · exact c6037.trans k0
    · exact c6038.trans k0
    · exact c6039.trans k0
    · exact c6040.trans k0
    · exact c6041.trans k0
    · exact c6042.trans k0
    · exact c6043.trans k0
    · exact c6044.trans k0
    · exact c6045.trans k0
    · exact c6046.trans k0
    · exact c6047.trans k0
    · exact c6048.trans k0
    · exact c6049.trans k0
    · exact c6050.trans k0
    · exact c6051.trans k0
    · exact c6052.trans k0
    · exact c6053.trans k0
    · exact c6054.trans k0
    · exact c6055.trans k0
    · exact c6056.trans k0
    · exact c6057.trans k0
    · exact c6058.trans k0
    · exact c6059.trans k0
    · exact c6060.trans k0
    · exact c6061.trans k0
    · exact c6062.trans k0
    · exact c6063.trans k0
    · exact c6064.trans k0
    · exact c6065.trans k0
    · exact c6066.trans k0
    · exact c6067.trans k0
    · exact c6068.trans k0
    · exact c6069.trans k0
    · exact c6070.trans k0
    · exact c6071.trans k0
    · exact c6072.trans k0
    · exact c6073.trans k0
    · exact c6074.trans k0
    · exact c6075.trans k0
    · exact c6076.trans k0
    · exact c6077.trans k0
    · exact c6078.trans k0
    · exact c6079.trans k0
    · exact c6080.trans k0
    · exact c6081.trans k0
    · exact c6082.trans k0
    )
  rwa [c6083] at hr

theorem leafCircuit_f1133 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1134 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = band (a (.virt 292)) (a (.wire 158 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1135 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = band (a (.wire 158 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1136 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = bor (a (.virt 14)) (a (.wire 158 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1137 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 158 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1138 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1139 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = a (.virt 14) + a (.wire 158 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1140 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = a (.wire 158 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1141 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 27) = bnot (a (.wire 158 5)) := by
  have c6084 := (leafCircuit_copies190 a h).2.2.2.2.1
  have c6085 := (leafCircuit_copies190 a h).2.2.2.2.2.1
  have c6086 := (leafCircuit_copies190 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_6 := arithEq_of_rows h (row := 157) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_6
  simp only [← c6084, k3, ← c6085, k3, ← c6086] at e_157_6
  have hr := e_157_6
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1142 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 27) = band (a (.virt 292)) (a (.wire 157 27)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1143 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1144 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 158 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1145 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 157 27)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1146 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = bor (a (.wire 158 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1147 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 4) = a (.virt 292) * a (.wire 158 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1148 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 27) = a (.virt 294) * a (.wire 158 4) := by
  have c6087 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.1
  have c6088 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.1
  have c6089 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_149_6 := arithEq_of_rows h (row := 149) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_6
  simp only [← c6087, k5, ← c6088, ← c6089, k5] at e_149_6
  have hr := e_149_6
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1149 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 35) = a (.virt 292) + a (.wire 158 4) := by
  have c6090 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6091 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6092 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_8 := arithEq_of_rows h (row := 154) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_8
  simp only [← c6090, k3, ← c6091, k3, ← c6092] at e_154_8
  have hr := e_154_8
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1150 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 31) = a (.wire 154 35) - a (.wire 149 27) := by
  have c6093 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6094 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6095 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_7 := arithEq_of_rows h (row := 157) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_7
  simp only [← c6093, ← c6094, k3, ← c6095] at e_157_7
  have hr := e_157_7
  linear_combination hr

theorem leafCircuit_f1151 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 35) = bnot (a (.wire 157 31)) := by
  have c6096 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6097 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6098 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_8 := arithEq_of_rows h (row := 157) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_8
  simp only [← c6096, k3, ← c6097, k3, ← c6098] at e_157_8
  have hr := e_157_8
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1152 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 31) = band (a (.wire 157 27)) (a (.wire 157 35)) := by
  have c6099 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6100 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6101 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_7 := arithEq_of_rows h (row := 149) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_7
  simp only [← c6099, ← c6100, ← c6101] at e_149_7
  have hr := e_149_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1153 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1154 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 158 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1155 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 149 31)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1156 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 5) = bor (a (.wire 158 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1157 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 3) = a (.virt 292) * a (.wire 158 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1158 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 35) = a (.virt 294) * a (.wire 158 3) := by
  have c6102 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6103 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6104 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_149_8 := arithEq_of_rows h (row := 149) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_8
  simp only [← c6102, k5, ← c6103, ← c6104, k5] at e_149_8
  have hr := e_149_8
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1159 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 39) = a (.virt 292) + a (.wire 158 3) := by
  have c6105 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6106 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6107 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_9 := arithEq_of_rows h (row := 154) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_9
  simp only [← c6105, k3, ← c6106, k3, ← c6107] at e_154_9
  have hr := e_154_9
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1160 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 39) = a (.wire 154 39) - a (.wire 149 35) := by
  have c6108 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6109 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6110 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_9 := arithEq_of_rows h (row := 157) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_9
  simp only [← c6108, ← c6109, k3, ← c6110] at e_157_9
  have hr := e_157_9
  linear_combination hr

theorem leafCircuit_f1161 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 43) = bnot (a (.wire 157 39)) := by
  have c6111 := (leafCircuit_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6112 := (leafCircuit_copies191 a h).1
  have c6113 := (leafCircuit_copies191 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_10 := arithEq_of_rows h (row := 157) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_10
  simp only [← c6111, k3, ← c6112, k3, ← c6113] at e_157_10
  have hr := e_157_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1162 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 39) = band (a (.wire 149 31)) (a (.wire 157 43)) := by
  have c6114 := (leafCircuit_copies191 a h).2.2.1
  have c6115 := (leafCircuit_copies191 a h).2.2.2.1
  have c6116 := (leafCircuit_copies191 a h).2.2.2.2.1
  have e_149_9 := arithEq_of_rows h (row := 149) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_9
  simp only [← c6114, ← c6115, ← c6116] at e_149_9
  have hr := e_149_9
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1163 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1164 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 2) = band (a (.virt 292)) (a (.wire 158 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1165 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 43) = band (a (.wire 158 2)) (a (.wire 149 39)) := by
  have c6117 := (leafCircuit_copies191 a h).2.2.2.2.2.1
  have c6118 := (leafCircuit_copies191 a h).2.2.2.2.2.2.1
  have c6119 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.1
  have e_149_10 := arithEq_of_rows h (row := 149) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_10
  simp only [← c6117, ← c6118, ← c6119] at e_149_10
  have hr := e_149_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1166 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 43) = bor (a (.wire 158 5)) (a (.wire 149 43)) := by
  have c6120 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.1
  have c6121 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.1
  have c6122 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6123 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6124 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6125 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_13 := arithEq_of_rows h (row := 104) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_13
  simp only [← c6120, ← c6121, ← c6122] at e_104_13
  have e_154_10 := arithEq_of_rows h (row := 154) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_10
  simp only [← c6123, ← c6124, k3, ← c6125] at e_154_10
  have hr := e_154_10
  simp only [e_104_13] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1167 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 158 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1168 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1169 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 2) = a (.virt 14) + a (.wire 158 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1170 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 2) = a (.wire 158 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1171 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 47) = bnot (a (.wire 158 2)) := by
  have c6126 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6127 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6128 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_11 := arithEq_of_rows h (row := 157) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_11
  simp only [← c6126, k3, ← c6127, k3, ← c6128] at e_157_11
  have hr := e_157_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1172 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 47) = band (a (.wire 149 39)) (a (.wire 157 47)) := by
  have c6129 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6130 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6131 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_11 := arithEq_of_rows h (row := 149) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_11
  simp only [← c6129, ← c6130, ← c6131] at e_149_11
  have hr := e_149_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1173 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1174 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 1) = band (a (.virt 292)) (a (.wire 158 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1175 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 51) = band (a (.wire 158 1)) (a (.wire 149 47)) := by
  have c6132 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6133 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6134 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_12 := arithEq_of_rows h (row := 149) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_12
  simp only [← c6132, ← c6133, ← c6134] at e_149_12
  have hr := e_149_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1176 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 47) = bor (a (.wire 154 43)) (a (.wire 149 51)) := by
  have c6135 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6136 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6137 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6138 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6139 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6140 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_104_14 := arithEq_of_rows h (row := 104) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_14
  simp only [← c6135, ← c6136, ← c6137] at e_104_14
  have e_154_11 := arithEq_of_rows h (row := 154) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_11
  simp only [← c6138, ← c6139, k3, ← c6140] at e_154_11
  have hr := e_154_11
  simp only [e_104_14] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1177 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 158 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1178 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1179 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 1) = a (.virt 14) + a (.wire 158 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1180 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 158 1) = a (.wire 158 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1181 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 157 51) = bnot (a (.wire 158 1)) := by
  have c6141 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6142 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6143 := (leafCircuit_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_157_12 := arithEq_of_rows h (row := 157) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_12
  simp only [← c6141, k3, ← c6142, k3, ← c6143] at e_157_12
  have hr := e_157_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1182 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 149 55) = band (a (.wire 149 47)) (a (.wire 157 51)) := by
  have c6144 := (leafCircuit_copies192 a h).1
  have c6145 := (leafCircuit_copies192 a h).2.1
  have c6146 := (leafCircuit_copies192 a h).2.2.1
  have e_149_13 := arithEq_of_rows h (row := 149) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_13
  simp only [← c6144, ← c6145, ← c6146] at e_149_13
  have hr := e_149_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1183 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 219)) 2 := by
  have c6147 := (leafCircuit_copies192 a h).2.2.2.1
  have c6148 := (leafCircuit_copies192 a h).2.2.2.2.1
  have c6149 := (leafCircuit_copies192 a h).2.2.2.2.2.1
  have c6150 := (leafCircuit_copies192 a h).2.2.2.2.2.2.1
  have c6151 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.1
  have c6152 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.1
  have c6153 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.1
  have c6154 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6155 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6156 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6157 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6158 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6159 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6160 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6161 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6162 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6163 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6164 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6165 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6166 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6167 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6168 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6169 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6170 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6171 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6172 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6173 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6174 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6175 := (leafCircuit_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6176 := (leafCircuit_copies193 a h).1
  have c6177 := (leafCircuit_copies193 a h).2.1
  have c6178 := (leafCircuit_copies193 a h).2.2.1
  have c6179 := (leafCircuit_copies193 a h).2.2.2.1
  have c6180 := (leafCircuit_copies193 a h).2.2.2.2.1
  have c6181 := (leafCircuit_copies193 a h).2.2.2.2.2.1
  have c6182 := (leafCircuit_copies193 a h).2.2.2.2.2.2.1
  have c6183 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.1
  have c6184 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.1
  have c6185 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.1
  have c6186 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6187 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6188 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6189 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6190 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6191 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6192 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6193 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6194 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6195 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6196 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6197 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6198 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6199 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6200 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6201 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6202 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6203 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6204 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 159) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c6147.trans k0
    · exact c6148.trans k0
    · exact c6149.trans k0
    · exact c6150.trans k0
    · exact c6151.trans k0
    · exact c6152.trans k0
    · exact c6153.trans k0
    · exact c6154.trans k0
    · exact c6155.trans k0
    · exact c6156.trans k0
    · exact c6157.trans k0
    · exact c6158.trans k0
    · exact c6159.trans k0
    · exact c6160.trans k0
    · exact c6161.trans k0
    · exact c6162.trans k0
    · exact c6163.trans k0
    · exact c6164.trans k0
    · exact c6165.trans k0
    · exact c6166.trans k0
    · exact c6167.trans k0
    · exact c6168.trans k0
    · exact c6169.trans k0
    · exact c6170.trans k0
    · exact c6171.trans k0
    · exact c6172.trans k0
    · exact c6173.trans k0
    · exact c6174.trans k0
    · exact c6175.trans k0
    · exact c6176.trans k0
    · exact c6177.trans k0
    · exact c6178.trans k0
    · exact c6179.trans k0
    · exact c6180.trans k0
    · exact c6181.trans k0
    · exact c6182.trans k0
    · exact c6183.trans k0
    · exact c6184.trans k0
    · exact c6185.trans k0
    · exact c6186.trans k0
    · exact c6187.trans k0
    · exact c6188.trans k0
    · exact c6189.trans k0
    · exact c6190.trans k0
    · exact c6191.trans k0
    · exact c6192.trans k0
    · exact c6193.trans k0
    · exact c6194.trans k0
    · exact c6195.trans k0
    · exact c6196.trans k0
    · exact c6197.trans k0
    · exact c6198.trans k0
    · exact c6199.trans k0
    · exact c6200.trans k0
    · exact c6201.trans k0
    · exact c6202.trans k0
    · exact c6203.trans k0
    )
  rwa [c6204] at hr

theorem leafCircuit_f1184 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 219)) (a (.virt 14)) (a (.virt 392)) (a (.virt 393)) := by
  have c6205 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6206 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6207 := (leafCircuit_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6208 := (leafCircuit_copies194 a h).1
  have c6209 := (leafCircuit_copies194 a h).2.1
  have c6210 := (leafCircuit_copies194 a h).2.2.1
  have c6211 := (leafCircuit_copies194 a h).2.2.2.1
  have c6212 := (leafCircuit_copies194 a h).2.2.2.2.1
  have c6213 := (leafCircuit_copies194 a h).2.2.2.2.2.1
  have c6214 := (leafCircuit_copies194 a h).2.2.2.2.2.2.1
  have c6215 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.1
  have c6216 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.1
  have c6217 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.1
  have c6218 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_149_14 := arithEq_of_rows h (row := 149) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_14
  simp only [← c6208, ← c6209, ← c6210] at e_149_14
  have e_157_13 := arithEq_of_rows h (row := 157) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_13
  simp only [← c6205, k3, ← c6206, k3, ← c6207] at e_157_13
  have e_157_14 := arithEq_of_rows h (row := 157) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_14
  simp only [← c6214, ← c6215, k3, ← c6216] at e_157_14
  have e_160_0 := arithEq_of_rows h (row := 160) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_0
  simp only [← c6211, ← c6212, ← c6213] at e_160_0
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_149_14
    linear_combination c6217.trans k0 - hc
  · have hc := e_157_14
    simp only [e_157_13, e_160_0] at hc
    linear_combination c6218.trans k0 - hc

theorem leafCircuit_f1185 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 219)) (a (.virt 292)) (a (.virt 394)) (a (.virt 395)) := by
  have c6219 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6220 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6221 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6222 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6223 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6224 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6225 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6226 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6227 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6228 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6229 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6230 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6231 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6232 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6233 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6234 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6235 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_160_1 := arithEq_of_rows h (row := 160) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_1
  simp only [← c6225, ← c6226, ← c6227] at e_160_1
  have e_160_2 := arithEq_of_rows h (row := 160) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_2
  simp only [← c6228, ← c6229, ← c6230] at e_160_2
  have e_161_0 := arithEq_of_rows h (row := 161) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_0
  simp only [← c6219, k3, ← c6220, k3, ← c6221] at e_161_0
  have e_161_1 := arithEq_of_rows h (row := 161) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_1
  simp only [← c6222, ← c6223, k3, ← c6224, k3] at e_161_1
  have e_161_2 := arithEq_of_rows h (row := 161) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_2
  simp only [← c6231, ← c6232, k3, ← c6233] at e_161_2
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_160_1
    simp only [e_161_1] at hc
    linear_combination c6234.trans k0 - hc
  · have hc := e_161_2
    simp only [e_161_0, e_160_2, e_161_1] at hc
    linear_combination c6235.trans k0 - hc

theorem leafCircuit_f1186 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 219)) (a (.virt 294)) (a (.virt 396)) (a (.virt 397)) := by
  have c6236 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6237 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6238 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6239 := (leafCircuit_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6240 := (leafCircuit_copies195 a h).1
  have c6241 := (leafCircuit_copies195 a h).2.1
  have c6242 := (leafCircuit_copies195 a h).2.2.1
  have c6243 := (leafCircuit_copies195 a h).2.2.2.1
  have c6244 := (leafCircuit_copies195 a h).2.2.2.2.1
  have c6245 := (leafCircuit_copies195 a h).2.2.2.2.2.1
  have c6246 := (leafCircuit_copies195 a h).2.2.2.2.2.2.1
  have c6247 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.1
  have c6248 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.1
  have c6249 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.1
  have c6250 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6251 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6252 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_160_3 := arithEq_of_rows h (row := 160) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_3
  simp only [← c6242, ← c6243, ← c6244] at e_160_3
  have e_160_4 := arithEq_of_rows h (row := 160) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_4
  simp only [← c6245, ← c6246, ← c6247] at e_160_4
  have e_161_3 := arithEq_of_rows h (row := 161) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_3
  simp only [← c6236, k3, ← c6237, k3, ← c6238] at e_161_3
  have e_161_4 := arithEq_of_rows h (row := 161) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_4
  simp only [← c6239, ← c6240, k3, ← c6241, k5] at e_161_4
  have e_161_5 := arithEq_of_rows h (row := 161) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_5
  simp only [← c6248, ← c6249, k3, ← c6250] at e_161_5
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_160_3
    simp only [e_161_4] at hc
    linear_combination c6251.trans k0 - hc
  · have hc := e_161_5
    simp only [e_161_3, e_160_4, e_161_4] at hc
    linear_combination c6252.trans k0 - hc

theorem leafCircuit_f1187 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 219)) (a (.virt 295)) (a (.virt 398)) (a (.virt 399)) := by
  have c6253 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6254 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6255 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6256 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6257 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6258 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6259 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6260 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6261 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6262 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6263 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6264 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6265 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6266 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6267 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6268 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6269 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_160_5 := arithEq_of_rows h (row := 160) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_5
  simp only [← c6259, ← c6260, ← c6261] at e_160_5
  have e_160_6 := arithEq_of_rows h (row := 160) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_6
  simp only [← c6262, ← c6263, ← c6264] at e_160_6
  have e_161_6 := arithEq_of_rows h (row := 161) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_6
  simp only [← c6253, k3, ← c6254, k3, ← c6255] at e_161_6
  have e_161_7 := arithEq_of_rows h (row := 161) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_7
  simp only [← c6256, ← c6257, k3, ← c6258, k6] at e_161_7
  have e_161_8 := arithEq_of_rows h (row := 161) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_8
  simp only [← c6265, ← c6266, k3, ← c6267] at e_161_8
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_160_5
    simp only [e_161_7] at hc
    linear_combination c6268.trans k0 - hc
  · have hc := e_161_8
    simp only [e_161_6, e_160_6, e_161_7] at hc
    linear_combination c6269.trans k0 - hc

theorem leafCircuit_f1188 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 161 43) = bselect (a (.virt 392)) (a (.wire 152 59)) (a (.virt 159)) := by
  have c6270 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6271 := (leafCircuit_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6272 := (leafCircuit_copies196 a h).1
  have c6273 := (leafCircuit_copies196 a h).2.1
  have c6274 := (leafCircuit_copies196 a h).2.2.1
  have c6275 := (leafCircuit_copies196 a h).2.2.2.1
  have e_161_9 := arithEq_of_rows h (row := 161) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_9
  simp only [← c6270, ← c6271, ← c6272] at e_161_9
  have e_161_10 := arithEq_of_rows h (row := 161) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_10
  simp only [← c6273, ← c6274, ← c6275] at e_161_10
  have hr := e_161_10
  simp only [e_161_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1189 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 161 51) = bselect (a (.virt 392)) (a (.wire 157 7)) (a (.virt 160)) := by
  have c6276 := (leafCircuit_copies196 a h).2.2.2.2.1
  have c6277 := (leafCircuit_copies196 a h).2.2.2.2.2.1
  have c6278 := (leafCircuit_copies196 a h).2.2.2.2.2.2.1
  have c6279 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.1
  have c6280 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.1
  have c6281 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.1
  have e_161_11 := arithEq_of_rows h (row := 161) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_11
  simp only [← c6276, ← c6277, ← c6278] at e_161_11
  have e_161_12 := arithEq_of_rows h (row := 161) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_12
  simp only [← c6279, ← c6280, ← c6281] at e_161_12
  have hr := e_161_12
  simp only [e_161_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1190 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 161 59) = bselect (a (.virt 392)) (a (.wire 157 15)) (a (.virt 161)) := by
  have c6282 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6283 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6284 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6285 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6286 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6287 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_13 := arithEq_of_rows h (row := 161) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_13
  simp only [← c6282, ← c6283, ← c6284] at e_161_13
  have e_161_14 := arithEq_of_rows h (row := 161) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_14
  simp only [← c6285, ← c6286, ← c6287] at e_161_14
  have hr := e_161_14
  simp only [e_161_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1191 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 7) = bselect (a (.virt 392)) (a (.wire 157 23)) (a (.virt 162)) := by
  have c6288 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6289 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6290 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6291 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6292 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6293 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_0 := arithEq_of_rows h (row := 162) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_0
  simp only [← c6288, ← c6289, ← c6290] at e_162_0
  have e_162_1 := arithEq_of_rows h (row := 162) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_1
  simp only [← c6291, ← c6292, ← c6293] at e_162_1
  have hr := e_162_1
  simp only [e_162_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1192 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 15) = bselect (a (.virt 392)) (a (.virt 159)) (a (.virt 163)) := by
  have c6294 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6295 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6296 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6297 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6298 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6299 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_2 := arithEq_of_rows h (row := 162) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_2
  simp only [← c6294, ← c6295, ← c6296] at e_162_2
  have e_162_3 := arithEq_of_rows h (row := 162) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_3
  simp only [← c6297, ← c6298, ← c6299] at e_162_3
  have hr := e_162_3
  simp only [e_162_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1193 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 23) = bselect (a (.virt 394)) (a (.wire 152 59)) (a (.wire 162 15)) := by
  have c6300 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6301 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6302 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6303 := (leafCircuit_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6304 := (leafCircuit_copies197 a h).1
  have c6305 := (leafCircuit_copies197 a h).2.1
  have e_162_4 := arithEq_of_rows h (row := 162) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_4
  simp only [← c6300, ← c6301, ← c6302] at e_162_4
  have e_162_5 := arithEq_of_rows h (row := 162) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_5
  simp only [← c6303, ← c6304, ← c6305] at e_162_5
  have hr := e_162_5
  simp only [e_162_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1194 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 31) = bselect (a (.virt 392)) (a (.virt 160)) (a (.virt 164)) := by
  have c6306 := (leafCircuit_copies197 a h).2.2.1
  have c6307 := (leafCircuit_copies197 a h).2.2.2.1
  have c6308 := (leafCircuit_copies197 a h).2.2.2.2.1
  have c6309 := (leafCircuit_copies197 a h).2.2.2.2.2.1
  have c6310 := (leafCircuit_copies197 a h).2.2.2.2.2.2.1
  have c6311 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.1
  have e_162_6 := arithEq_of_rows h (row := 162) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_6
  simp only [← c6306, ← c6307, ← c6308] at e_162_6
  have e_162_7 := arithEq_of_rows h (row := 162) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_7
  simp only [← c6309, ← c6310, ← c6311] at e_162_7
  have hr := e_162_7
  simp only [e_162_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1195 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 39) = bselect (a (.virt 394)) (a (.wire 157 7)) (a (.wire 162 31)) := by
  have c6312 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.1
  have c6313 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.1
  have c6314 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6315 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6316 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6317 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_8 := arithEq_of_rows h (row := 162) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_8
  simp only [← c6312, ← c6313, ← c6314] at e_162_8
  have e_162_9 := arithEq_of_rows h (row := 162) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_9
  simp only [← c6315, ← c6316, ← c6317] at e_162_9
  have hr := e_162_9
  simp only [e_162_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1196 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 47) = bselect (a (.virt 392)) (a (.virt 161)) (a (.virt 165)) := by
  have c6318 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6319 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6320 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6321 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6322 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6323 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_10 := arithEq_of_rows h (row := 162) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_10
  simp only [← c6318, ← c6319, ← c6320] at e_162_10
  have e_162_11 := arithEq_of_rows h (row := 162) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_11
  simp only [← c6321, ← c6322, ← c6323] at e_162_11
  have hr := e_162_11
  simp only [e_162_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1197 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 162 55) = bselect (a (.virt 394)) (a (.wire 157 15)) (a (.wire 162 47)) := by
  have c6324 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6325 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6326 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6327 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6328 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6329 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_12 := arithEq_of_rows h (row := 162) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_12
  simp only [← c6324, ← c6325, ← c6326] at e_162_12
  have e_162_13 := arithEq_of_rows h (row := 162) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_13
  simp only [← c6327, ← c6328, ← c6329] at e_162_13
  have hr := e_162_13
  simp only [e_162_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1198 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 3) = bselect (a (.virt 392)) (a (.virt 162)) (a (.virt 166)) := by
  have c6330 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6331 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6332 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6333 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6334 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6335 := (leafCircuit_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_162_14 := arithEq_of_rows h (row := 162) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_14
  simp only [← c6330, ← c6331, ← c6332] at e_162_14
  have e_163_0 := arithEq_of_rows h (row := 163) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_0
  simp only [← c6333, ← c6334, ← c6335] at e_163_0
  have hr := e_163_0
  simp only [e_162_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1199 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 11) = bselect (a (.virt 394)) (a (.wire 157 23)) (a (.wire 163 3)) := by
  have c6336 := (leafCircuit_copies198 a h).1
  have c6337 := (leafCircuit_copies198 a h).2.1
  have c6338 := (leafCircuit_copies198 a h).2.2.1
  have c6339 := (leafCircuit_copies198 a h).2.2.2.1
  have c6340 := (leafCircuit_copies198 a h).2.2.2.2.1
  have c6341 := (leafCircuit_copies198 a h).2.2.2.2.2.1
  have e_163_1 := arithEq_of_rows h (row := 163) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_1
  simp only [← c6336, ← c6337, ← c6338] at e_163_1
  have e_163_2 := arithEq_of_rows h (row := 163) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_2
  simp only [← c6339, ← c6340, ← c6341] at e_163_2
  have hr := e_163_2
  simp only [e_163_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1200 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) := by
  have c6342 := (leafCircuit_copies198 a h).2.2.2.2.2.2.1
  have c6343 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.1
  have c6344 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.1
  have c6345 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.1
  have c6346 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6347 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_12 := arithEq_of_rows h (row := 154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_12
  simp only [← c6345, ← c6346, k3, ← c6347] at e_154_12
  have e_164_0 := arithEq_of_rows h (row := 164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_0
  simp only [← c6342, ← c6343, ← c6344] at e_164_0
  have hr := e_154_12
  simp only [e_164_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1201 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 19) = bselect (a (.wire 154 51)) (a (.virt 163)) (a (.virt 167)) := by
  have c6348 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6349 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6350 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6351 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6352 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6353 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_3 := arithEq_of_rows h (row := 163) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_3
  simp only [← c6348, ← c6349, ← c6350] at e_163_3
  have e_163_4 := arithEq_of_rows h (row := 163) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_4
  simp only [← c6351, ← c6352, ← c6353] at e_163_4
  have hr := e_163_4
  simp only [e_163_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1202 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 27) = bselect (a (.virt 396)) (a (.wire 152 59)) (a (.wire 163 19)) := by
  have c6354 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6355 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6356 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6357 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6358 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6359 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_5 := arithEq_of_rows h (row := 163) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_5
  simp only [← c6354, ← c6355, ← c6356] at e_163_5
  have e_163_6 := arithEq_of_rows h (row := 163) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_6
  simp only [← c6357, ← c6358, ← c6359] at e_163_6
  have hr := e_163_6
  simp only [e_163_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1203 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) := by
  have c6342 := (leafCircuit_copies198 a h).2.2.2.2.2.2.1
  have c6343 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.1
  have c6344 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.1
  have c6345 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.1
  have c6346 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6347 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_12 := arithEq_of_rows h (row := 154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_12
  simp only [← c6345, ← c6346, k3, ← c6347] at e_154_12
  have e_164_0 := arithEq_of_rows h (row := 164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_0
  simp only [← c6342, ← c6343, ← c6344] at e_164_0
  have hr := e_154_12
  simp only [e_164_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1204 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 35) = bselect (a (.wire 154 51)) (a (.virt 164)) (a (.virt 168)) := by
  have c6360 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6361 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6362 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6363 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6364 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6365 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_7 := arithEq_of_rows h (row := 163) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_7
  simp only [← c6360, ← c6361, ← c6362] at e_163_7
  have e_163_8 := arithEq_of_rows h (row := 163) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_8
  simp only [← c6363, ← c6364, ← c6365] at e_163_8
  have hr := e_163_8
  simp only [e_163_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1205 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 43) = bselect (a (.virt 396)) (a (.wire 157 7)) (a (.wire 163 35)) := by
  have c6366 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6367 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6368 := (leafCircuit_copies199 a h).1
  have c6369 := (leafCircuit_copies199 a h).2.1
  have c6370 := (leafCircuit_copies199 a h).2.2.1
  have c6371 := (leafCircuit_copies199 a h).2.2.2.1
  have e_163_9 := arithEq_of_rows h (row := 163) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_9
  simp only [← c6366, ← c6367, ← c6368] at e_163_9
  have e_163_10 := arithEq_of_rows h (row := 163) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_10
  simp only [← c6369, ← c6370, ← c6371] at e_163_10
  have hr := e_163_10
  simp only [e_163_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1206 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) := by
  have c6342 := (leafCircuit_copies198 a h).2.2.2.2.2.2.1
  have c6343 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.1
  have c6344 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.1
  have c6345 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.1
  have c6346 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6347 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_12 := arithEq_of_rows h (row := 154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_12
  simp only [← c6345, ← c6346, k3, ← c6347] at e_154_12
  have e_164_0 := arithEq_of_rows h (row := 164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_0
  simp only [← c6342, ← c6343, ← c6344] at e_164_0
  have hr := e_154_12
  simp only [e_164_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1207 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 51) = bselect (a (.wire 154 51)) (a (.virt 165)) (a (.virt 169)) := by
  have c6372 := (leafCircuit_copies199 a h).2.2.2.2.1
  have c6373 := (leafCircuit_copies199 a h).2.2.2.2.2.1
  have c6374 := (leafCircuit_copies199 a h).2.2.2.2.2.2.1
  have c6375 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.1
  have c6376 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.1
  have c6377 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.1
  have e_163_11 := arithEq_of_rows h (row := 163) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_11
  simp only [← c6372, ← c6373, ← c6374] at e_163_11
  have e_163_12 := arithEq_of_rows h (row := 163) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_12
  simp only [← c6375, ← c6376, ← c6377] at e_163_12
  have hr := e_163_12
  simp only [e_163_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1208 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 163 59) = bselect (a (.virt 396)) (a (.wire 157 15)) (a (.wire 163 51)) := by
  have c6378 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6379 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6380 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6381 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6382 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6383 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_13 := arithEq_of_rows h (row := 163) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_13
  simp only [← c6378, ← c6379, ← c6380] at e_163_13
  have e_163_14 := arithEq_of_rows h (row := 163) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_14
  simp only [← c6381, ← c6382, ← c6383] at e_163_14
  have hr := e_163_14
  simp only [e_163_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1209 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) := by
  have c6342 := (leafCircuit_copies198 a h).2.2.2.2.2.2.1
  have c6343 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.1
  have c6344 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.1
  have c6345 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.1
  have c6346 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6347 := (leafCircuit_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_12 := arithEq_of_rows h (row := 154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_12
  simp only [← c6345, ← c6346, k3, ← c6347] at e_154_12
  have e_164_0 := arithEq_of_rows h (row := 164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_0
  simp only [← c6342, ← c6343, ← c6344] at e_164_0
  have hr := e_154_12
  simp only [e_164_0] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1210 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 7) = bselect (a (.wire 154 51)) (a (.virt 166)) (a (.virt 170)) := by
  have c6384 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6385 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6386 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6387 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6388 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6389 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_0 := arithEq_of_rows h (row := 165) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_0
  simp only [← c6384, ← c6385, ← c6386] at e_165_0
  have e_165_1 := arithEq_of_rows h (row := 165) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_1
  simp only [← c6387, ← c6388, ← c6389] at e_165_1
  have hr := e_165_1
  simp only [e_165_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1211 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 15) = bselect (a (.virt 396)) (a (.wire 157 23)) (a (.wire 165 7)) := by
  have c6390 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6391 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6392 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6393 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6394 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6395 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_2 := arithEq_of_rows h (row := 165) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_2
  simp only [← c6390, ← c6391, ← c6392] at e_165_2
  have e_165_3 := arithEq_of_rows h (row := 165) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_3
  simp only [← c6393, ← c6394, ← c6395] at e_165_3
  have hr := e_165_3
  simp only [e_165_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1212 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 23) = bselect (a (.virt 398)) (a (.wire 152 59)) (a (.virt 167)) := by
  have c6396 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6397 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6398 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6399 := (leafCircuit_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6400 := (leafCircuit_copies200 a h).1
  have c6401 := (leafCircuit_copies200 a h).2.1
  have e_165_4 := arithEq_of_rows h (row := 165) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_4
  simp only [← c6396, ← c6397, ← c6398] at e_165_4
  have e_165_5 := arithEq_of_rows h (row := 165) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_5
  simp only [← c6399, ← c6400, ← c6401] at e_165_5
  have hr := e_165_5
  simp only [e_165_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1213 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 31) = bselect (a (.virt 398)) (a (.wire 157 7)) (a (.virt 168)) := by
  have c6402 := (leafCircuit_copies200 a h).2.2.1
  have c6403 := (leafCircuit_copies200 a h).2.2.2.1
  have c6404 := (leafCircuit_copies200 a h).2.2.2.2.1
  have c6405 := (leafCircuit_copies200 a h).2.2.2.2.2.1
  have c6406 := (leafCircuit_copies200 a h).2.2.2.2.2.2.1
  have c6407 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.1
  have e_165_6 := arithEq_of_rows h (row := 165) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_6
  simp only [← c6402, ← c6403, ← c6404] at e_165_6
  have e_165_7 := arithEq_of_rows h (row := 165) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_7
  simp only [← c6405, ← c6406, ← c6407] at e_165_7
  have hr := e_165_7
  simp only [e_165_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1214 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 39) = bselect (a (.virt 398)) (a (.wire 157 15)) (a (.virt 169)) := by
  have c6408 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.1
  have c6409 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.1
  have c6410 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6411 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6412 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6413 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_8 := arithEq_of_rows h (row := 165) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_8
  simp only [← c6408, ← c6409, ← c6410] at e_165_8
  have e_165_9 := arithEq_of_rows h (row := 165) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_9
  simp only [← c6411, ← c6412, ← c6413] at e_165_9
  have hr := e_165_9
  simp only [e_165_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1215 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 47) = bselect (a (.virt 398)) (a (.wire 157 23)) (a (.virt 170)) := by
  have c6414 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6415 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6416 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6417 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6418 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6419 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_10 := arithEq_of_rows h (row := 165) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_10
  simp only [← c6414, ← c6415, ← c6416] at e_165_10
  have e_165_11 := arithEq_of_rows h (row := 165) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_11
  simp only [← c6417, ← c6418, ← c6419] at e_165_11
  have hr := e_165_11
  simp only [e_165_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1216 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 169 12) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 0 ∧ a (.wire 169 13) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 1 ∧ a (.wire 169 14) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 2 ∧ a (.wire 169 15) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 3) := by
  have c6420 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6421 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6422 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6423 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6424 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6425 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6426 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6427 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6428 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6429 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6430 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6431 := (leafCircuit_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6432 := (leafCircuit_copies201 a h).1
  have c6433 := (leafCircuit_copies201 a h).2.1
  have c6434 := (leafCircuit_copies201 a h).2.2.1
  have c6435 := (leafCircuit_copies201 a h).2.2.2.1
  have c6436 := (leafCircuit_copies201 a h).2.2.2.2.1
  have c6437 := (leafCircuit_copies201 a h).2.2.2.2.2.1
  have c6438 := (leafCircuit_copies201 a h).2.2.2.2.2.2.1
  have c6439 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.1
  have c6440 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.1
  have c6441 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.1
  have c6442 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6443 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6444 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6445 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6446 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6447 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6448 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6449 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6450 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6451 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6452 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6453 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6454 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6455 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6456 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6457 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6458 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6459 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6460 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6461 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6462 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6463 := (leafCircuit_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6464 := (leafCircuit_copies202 a h).1
  have c6465 := (leafCircuit_copies202 a h).2.1
  have c6466 := (leafCircuit_copies202 a h).2.2.1
  have c6467 := (leafCircuit_copies202 a h).2.2.2.1
  have c6468 := (leafCircuit_copies202 a h).2.2.2.2.1
  have c6469 := (leafCircuit_copies202 a h).2.2.2.2.2.1
  have c6470 := (leafCircuit_copies202 a h).2.2.2.2.2.2.1
  have c6471 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.1
  have c6472 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.1
  have c6473 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.1
  have c6474 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6475 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6476 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6477 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6478 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6479 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6480 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6481 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6482 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_154_13 := arithEq_of_rows h (row := 154) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_13
  simp only [← c6432, ← c6433, k3, ← c6434] at e_154_13
  have e_154_14 := arithEq_of_rows h (row := 154) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_14
  simp only [← c6435, ← c6436, k3, ← c6437] at e_154_14
  have e_167_0 := arithEq_of_rows h (row := 167) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_0
  simp only [← c6438, ← c6439, k3, ← c6440] at e_167_0
  have e_167_1 := arithEq_of_rows h (row := 167) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_1
  simp only [← c6441, ← c6442, k3, ← c6443] at e_167_1
  have e_167_2 := arithEq_of_rows h (row := 167) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_2
  simp only [← c6444, ← c6445, k3, ← c6446] at e_167_2
  have e_167_3 := arithEq_of_rows h (row := 167) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_3
  simp only [← c6447, ← c6448, k3, ← c6449] at e_167_3
  have e_167_4 := arithEq_of_rows h (row := 167) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_4
  simp only [← c6450, ← c6451, k3, ← c6452] at e_167_4
  have e_167_5 := arithEq_of_rows h (row := 167) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_5
  simp only [← c6453, ← c6454, k3, ← c6455] at e_167_5
  have e_167_6 := arithEq_of_rows h (row := 167) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_6
  simp only [← c6468, ← c6469, k3, ← c6470, k3] at e_167_6
  have hin0 : poseidon2In a 166 = addBlock (fun _ => 0) [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11)] :=
    poseidon2In_first c6420.symm c6421.symm c6422.symm c6423.symm c6424.symm c6425.symm c6426.symm c6427.symm (c6428.symm.trans k0) (c6429.symm.trans k0) (c6430.symm.trans k0) (c6431.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 166) rfl rfl hin0
  have hin1 : poseidon2In a 168 = addBlock (poseidon2Out a 166) [a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] :=
    poseidon2In_chain (by linear_combination -c6456 + e_154_13) (by linear_combination -c6457 + e_154_14) (by linear_combination -c6458 + e_167_0) (by linear_combination -c6459 + e_167_1) (by linear_combination -c6460 + e_167_2) (by linear_combination -c6461 + e_167_3) (by linear_combination -c6462 + e_167_4) (by linear_combination -c6463 + e_167_5) c6464.symm c6465.symm c6466.symm c6467.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 168) rfl rfl hin1
  have hin2 : poseidon2In a 169 = addBlock (poseidon2Out a 168) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c6471 + e_167_6) (c6472.symm.trans (add_zero _).symm) (c6473.symm.trans (add_zero _).symm) (c6474.symm.trans (add_zero _).symm) (c6475.symm.trans (add_zero _).symm) (c6476.symm.trans (add_zero _).symm) (c6477.symm.trans (add_zero _).symm) (c6478.symm.trans (add_zero _).symm) c6479.symm c6480.symm c6481.symm c6482.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 169) rfl rfl hin2
  have hpad : pad10 [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] =
      [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11)] ++ ([a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1217 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 165 55) = bselect (a (.wire 154 47)) (a (.wire 169 12)) (a (.wire 152 59)) := by
  have c6483 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6484 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6485 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6486 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6487 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6488 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_12 := arithEq_of_rows h (row := 165) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_12
  simp only [← c6483, ← c6484, ← c6485] at e_165_12
  have e_165_13 := arithEq_of_rows h (row := 165) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_13
  simp only [← c6486, ← c6487, ← c6488] at e_165_13
  have hr := e_165_13
  simp only [e_165_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1218 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 3) = bselect (a (.wire 154 47)) (a (.wire 169 13)) (a (.wire 157 7)) := by
  have c6489 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6490 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6491 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6492 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6493 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6494 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_14 := arithEq_of_rows h (row := 165) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_14
  simp only [← c6489, ← c6490, ← c6491] at e_165_14
  have e_170_0 := arithEq_of_rows h (row := 170) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_0
  simp only [← c6492, ← c6493, ← c6494] at e_170_0
  have hr := e_170_0
  simp only [e_165_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1219 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 11) = bselect (a (.wire 154 47)) (a (.wire 169 14)) (a (.wire 157 15)) := by
  have c6495 := (leafCircuit_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6496 := (leafCircuit_copies203 a h).1
  have c6497 := (leafCircuit_copies203 a h).2.1
  have c6498 := (leafCircuit_copies203 a h).2.2.1
  have c6499 := (leafCircuit_copies203 a h).2.2.2.1
  have c6500 := (leafCircuit_copies203 a h).2.2.2.2.1
  have e_170_1 := arithEq_of_rows h (row := 170) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_1
  simp only [← c6495, ← c6496, ← c6497] at e_170_1
  have e_170_2 := arithEq_of_rows h (row := 170) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_2
  simp only [← c6498, ← c6499, ← c6500] at e_170_2
  have hr := e_170_2
  simp only [e_170_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1220 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 19) = bselect (a (.wire 154 47)) (a (.wire 169 15)) (a (.wire 157 23)) := by
  have c6501 := (leafCircuit_copies203 a h).2.2.2.2.2.1
  have c6502 := (leafCircuit_copies203 a h).2.2.2.2.2.2.1
  have c6503 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.1
  have c6504 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.1
  have c6505 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.1
  have c6506 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_170_3 := arithEq_of_rows h (row := 170) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_3
  simp only [← c6501, ← c6502, ← c6503] at e_170_3
  have e_170_4 := arithEq_of_rows h (row := 170) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_4
  simp only [← c6504, ← c6505, ← c6506] at e_170_4
  have hr := e_170_4
  simp only [e_170_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1221 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 171 1), a (.wire 171 2), a (.wire 171 3), a (.wire 171 4), a (.wire 171 5)] := by
  have c6507 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6508 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6509 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6510 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6511 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6512 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6513 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6514 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6515 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6516 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6517 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6518 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6519 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6520 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6521 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6522 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6523 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6524 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6525 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6526 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6527 := (leafCircuit_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6528 := (leafCircuit_copies204 a h).1
  have c6529 := (leafCircuit_copies204 a h).2.1
  have c6530 := (leafCircuit_copies204 a h).2.2.1
  have c6531 := (leafCircuit_copies204 a h).2.2.2.1
  have c6532 := (leafCircuit_copies204 a h).2.2.2.2.1
  have c6533 := (leafCircuit_copies204 a h).2.2.2.2.2.1
  have c6534 := (leafCircuit_copies204 a h).2.2.2.2.2.2.1
  have c6535 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.1
  have c6536 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.1
  have c6537 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.1
  have c6538 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6539 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6540 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6541 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6542 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6543 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6544 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6545 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6546 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6547 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6548 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6549 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6550 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6551 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6552 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6553 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6554 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6555 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6556 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6557 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6558 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6559 := (leafCircuit_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6560 := (leafCircuit_copies205 a h).1
  have c6561 := (leafCircuit_copies205 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 171) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c6507.trans k0
    · exact c6508.trans k0
    · exact c6509.trans k0
    · exact c6510.trans k0
    · exact c6511.trans k0
    · exact c6512.trans k0
    · exact c6513.trans k0
    · exact c6514.trans k0
    · exact c6515.trans k0
    · exact c6516.trans k0
    · exact c6517.trans k0
    · exact c6518.trans k0
    · exact c6519.trans k0
    · exact c6520.trans k0
    · exact c6521.trans k0
    · exact c6522.trans k0
    · exact c6523.trans k0
    · exact c6524.trans k0
    · exact c6525.trans k0
    · exact c6526.trans k0
    · exact c6527.trans k0
    · exact c6528.trans k0
    · exact c6529.trans k0
    · exact c6530.trans k0
    · exact c6531.trans k0
    · exact c6532.trans k0
    · exact c6533.trans k0
    · exact c6534.trans k0
    · exact c6535.trans k0
    · exact c6536.trans k0
    · exact c6537.trans k0
    · exact c6538.trans k0
    · exact c6539.trans k0
    · exact c6540.trans k0
    · exact c6541.trans k0
    · exact c6542.trans k0
    · exact c6543.trans k0
    · exact c6544.trans k0
    · exact c6545.trans k0
    · exact c6546.trans k0
    · exact c6547.trans k0
    · exact c6548.trans k0
    · exact c6549.trans k0
    · exact c6550.trans k0
    · exact c6551.trans k0
    · exact c6552.trans k0
    · exact c6553.trans k0
    · exact c6554.trans k0
    · exact c6555.trans k0
    · exact c6556.trans k0
    · exact c6557.trans k0
    · exact c6558.trans k0
    · exact c6559.trans k0
    · exact c6560.trans k0
    )
  rwa [c6561] at hr

theorem leafCircuit_f1222 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1223 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = band (a (.virt 292)) (a (.wire 171 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1224 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = band (a (.wire 171 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1225 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = bor (a (.virt 14)) (a (.wire 171 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1226 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 171 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1227 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1228 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = a (.virt 14) + a (.wire 171 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1229 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = a (.wire 171 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1230 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 23) = bnot (a (.wire 171 5)) := by
  have c6562 := (leafCircuit_copies205 a h).2.2.1
  have c6563 := (leafCircuit_copies205 a h).2.2.2.1
  have c6564 := (leafCircuit_copies205 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_5 := arithEq_of_rows h (row := 170) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_5
  simp only [← c6562, k3, ← c6563, k3, ← c6564] at e_170_5
  have hr := e_170_5
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1231 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 23) = band (a (.virt 292)) (a (.wire 170 23)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1232 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1233 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 171 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1234 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 170 23)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1235 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = bor (a (.wire 171 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1236 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 4) = a (.virt 292) * a (.wire 171 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1237 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 31) = a (.virt 294) * a (.wire 171 4) := by
  have c6565 := (leafCircuit_copies205 a h).2.2.2.2.2.1
  have c6566 := (leafCircuit_copies205 a h).2.2.2.2.2.2.1
  have c6567 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_160_7 := arithEq_of_rows h (row := 160) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_7
  simp only [← c6565, k5, ← c6566, ← c6567, k5] at e_160_7
  have hr := e_160_7
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1238 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 31) = a (.virt 292) + a (.wire 171 4) := by
  have c6568 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.1
  have c6569 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.1
  have c6570 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_167_7 := arithEq_of_rows h (row := 167) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_7
  simp only [← c6568, k3, ← c6569, k3, ← c6570] at e_167_7
  have hr := e_167_7
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1239 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 27) = a (.wire 167 31) - a (.wire 160 31) := by
  have c6571 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6572 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6573 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_6 := arithEq_of_rows h (row := 170) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_6
  simp only [← c6571, ← c6572, k3, ← c6573] at e_170_6
  have hr := e_170_6
  linear_combination hr

theorem leafCircuit_f1240 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 31) = bnot (a (.wire 170 27)) := by
  have c6574 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6575 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6576 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_7 := arithEq_of_rows h (row := 170) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_7
  simp only [← c6574, k3, ← c6575, k3, ← c6576] at e_170_7
  have hr := e_170_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1241 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 35) = band (a (.wire 170 23)) (a (.wire 170 31)) := by
  have c6577 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6578 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6579 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_8 := arithEq_of_rows h (row := 160) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_8
  simp only [← c6577, ← c6578, ← c6579] at e_160_8
  have hr := e_160_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1242 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1243 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 171 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1244 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 160 35)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1245 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 5) = bor (a (.wire 171 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1246 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 3) = a (.virt 292) * a (.wire 171 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1247 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 39) = a (.virt 294) * a (.wire 171 3) := by
  have c6580 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6581 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6582 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_160_9 := arithEq_of_rows h (row := 160) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_9
  simp only [← c6580, k5, ← c6581, ← c6582, k5] at e_160_9
  have hr := e_160_9
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1248 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 35) = a (.virt 292) + a (.wire 171 3) := by
  have c6583 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6584 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6585 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_167_8 := arithEq_of_rows h (row := 167) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_8
  simp only [← c6583, k3, ← c6584, k3, ← c6585] at e_167_8
  have hr := e_167_8
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1249 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 35) = a (.wire 167 35) - a (.wire 160 39) := by
  have c6586 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6587 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6588 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_8 := arithEq_of_rows h (row := 170) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_8
  simp only [← c6586, ← c6587, k3, ← c6588] at e_170_8
  have hr := e_170_8
  linear_combination hr

theorem leafCircuit_f1250 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 39) = bnot (a (.wire 170 35)) := by
  have c6589 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6590 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6591 := (leafCircuit_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_9 := arithEq_of_rows h (row := 170) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_9
  simp only [← c6589, k3, ← c6590, k3, ← c6591] at e_170_9
  have hr := e_170_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1251 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 43) = band (a (.wire 160 35)) (a (.wire 170 39)) := by
  have c6592 := (leafCircuit_copies206 a h).1
  have c6593 := (leafCircuit_copies206 a h).2.1
  have c6594 := (leafCircuit_copies206 a h).2.2.1
  have e_160_10 := arithEq_of_rows h (row := 160) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_10
  simp only [← c6592, ← c6593, ← c6594] at e_160_10
  have hr := e_160_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1252 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1253 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 2) = band (a (.virt 292)) (a (.wire 171 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1254 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 47) = band (a (.wire 171 2)) (a (.wire 160 43)) := by
  have c6595 := (leafCircuit_copies206 a h).2.2.2.1
  have c6596 := (leafCircuit_copies206 a h).2.2.2.2.1
  have c6597 := (leafCircuit_copies206 a h).2.2.2.2.2.1
  have e_160_11 := arithEq_of_rows h (row := 160) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_11
  simp only [← c6595, ← c6596, ← c6597] at e_160_11
  have hr := e_160_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1255 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 39) = bor (a (.wire 171 5)) (a (.wire 160 47)) := by
  have c6598 := (leafCircuit_copies206 a h).2.2.2.2.2.2.1
  have c6599 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.1
  have c6600 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.1
  have c6601 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.1
  have c6602 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6603 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_1 := arithEq_of_rows h (row := 164) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_1
  simp only [← c6598, ← c6599, ← c6600] at e_164_1
  have e_167_9 := arithEq_of_rows h (row := 167) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_9
  simp only [← c6601, ← c6602, k3, ← c6603] at e_167_9
  have hr := e_167_9
  simp only [e_164_1] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1256 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 171 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1257 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1258 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 2) = a (.virt 14) + a (.wire 171 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1259 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 2) = a (.wire 171 2) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1260 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 43) = bnot (a (.wire 171 2)) := by
  have c6604 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6605 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6606 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_10 := arithEq_of_rows h (row := 170) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_10
  simp only [← c6604, k3, ← c6605, k3, ← c6606] at e_170_10
  have hr := e_170_10
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1261 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 51) = band (a (.wire 160 43)) (a (.wire 170 43)) := by
  have c6607 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6608 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6609 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_12 := arithEq_of_rows h (row := 160) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_12
  simp only [← c6607, ← c6608, ← c6609] at e_160_12
  have hr := e_160_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1262 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1263 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 171 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1264 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 160 51)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1265 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 39) = bor (a (.wire 167 39)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1266 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 171 1) = a (.virt 292) * a (.wire 171 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1267 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 55) = a (.virt 294) * a (.wire 171 1) := by
  have c6610 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6611 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6612 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_160_13 := arithEq_of_rows h (row := 160) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_13
  simp only [← c6610, k5, ← c6611, ← c6612, k5] at e_160_13
  have hr := e_160_13
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1268 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 43) = a (.virt 292) + a (.wire 171 1) := by
  have c6613 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6614 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6615 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_167_10 := arithEq_of_rows h (row := 167) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_10
  simp only [← c6613, k3, ← c6614, k3, ← c6615] at e_167_10
  have hr := e_167_10
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1269 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 47) = a (.wire 167 43) - a (.wire 160 55) := by
  have c6616 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6617 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6618 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_11 := arithEq_of_rows h (row := 170) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_11
  simp only [← c6616, ← c6617, k3, ← c6618] at e_170_11
  have hr := e_170_11
  linear_combination hr

theorem leafCircuit_f1270 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 170 51) = bnot (a (.wire 170 47)) := by
  have c6619 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6620 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6621 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_12 := arithEq_of_rows h (row := 170) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_12
  simp only [← c6619, k3, ← c6620, k3, ← c6621] at e_170_12
  have hr := e_170_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1271 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 160 59) = band (a (.wire 160 51)) (a (.wire 170 51)) := by
  have c6622 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6623 := (leafCircuit_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6624 := (leafCircuit_copies207 a h).1
  have e_160_14 := arithEq_of_rows h (row := 160) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_14
  simp only [← c6622, ← c6623, ← c6624] at e_160_14
  have hr := e_160_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1272 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 220)) 2 := by
  have c6625 := (leafCircuit_copies207 a h).2.1
  have c6626 := (leafCircuit_copies207 a h).2.2.1
  have c6627 := (leafCircuit_copies207 a h).2.2.2.1
  have c6628 := (leafCircuit_copies207 a h).2.2.2.2.1
  have c6629 := (leafCircuit_copies207 a h).2.2.2.2.2.1
  have c6630 := (leafCircuit_copies207 a h).2.2.2.2.2.2.1
  have c6631 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.1
  have c6632 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.1
  have c6633 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.1
  have c6634 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6635 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6636 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6637 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6638 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6639 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6640 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6641 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6642 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6643 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6644 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6645 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6646 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6647 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6648 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6649 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6650 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6651 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6652 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6653 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6654 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6655 := (leafCircuit_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6656 := (leafCircuit_copies208 a h).1
  have c6657 := (leafCircuit_copies208 a h).2.1
  have c6658 := (leafCircuit_copies208 a h).2.2.1
  have c6659 := (leafCircuit_copies208 a h).2.2.2.1
  have c6660 := (leafCircuit_copies208 a h).2.2.2.2.1
  have c6661 := (leafCircuit_copies208 a h).2.2.2.2.2.1
  have c6662 := (leafCircuit_copies208 a h).2.2.2.2.2.2.1
  have c6663 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.1
  have c6664 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.1
  have c6665 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.1
  have c6666 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6667 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6668 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6669 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6670 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6671 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6672 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6673 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6674 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6675 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6676 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6677 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6678 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6679 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6680 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6681 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6682 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 172) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c6625.trans k0
    · exact c6626.trans k0
    · exact c6627.trans k0
    · exact c6628.trans k0
    · exact c6629.trans k0
    · exact c6630.trans k0
    · exact c6631.trans k0
    · exact c6632.trans k0
    · exact c6633.trans k0
    · exact c6634.trans k0
    · exact c6635.trans k0
    · exact c6636.trans k0
    · exact c6637.trans k0
    · exact c6638.trans k0
    · exact c6639.trans k0
    · exact c6640.trans k0
    · exact c6641.trans k0
    · exact c6642.trans k0
    · exact c6643.trans k0
    · exact c6644.trans k0
    · exact c6645.trans k0
    · exact c6646.trans k0
    · exact c6647.trans k0
    · exact c6648.trans k0
    · exact c6649.trans k0
    · exact c6650.trans k0
    · exact c6651.trans k0
    · exact c6652.trans k0
    · exact c6653.trans k0
    · exact c6654.trans k0
    · exact c6655.trans k0
    · exact c6656.trans k0
    · exact c6657.trans k0
    · exact c6658.trans k0
    · exact c6659.trans k0
    · exact c6660.trans k0
    · exact c6661.trans k0
    · exact c6662.trans k0
    · exact c6663.trans k0
    · exact c6664.trans k0
    · exact c6665.trans k0
    · exact c6666.trans k0
    · exact c6667.trans k0
    · exact c6668.trans k0
    · exact c6669.trans k0
    · exact c6670.trans k0
    · exact c6671.trans k0
    · exact c6672.trans k0
    · exact c6673.trans k0
    · exact c6674.trans k0
    · exact c6675.trans k0
    · exact c6676.trans k0
    · exact c6677.trans k0
    · exact c6678.trans k0
    · exact c6679.trans k0
    · exact c6680.trans k0
    · exact c6681.trans k0
    )
  rwa [c6682] at hr

theorem leafCircuit_f1273 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 220)) (a (.virt 14)) (a (.virt 400)) (a (.virt 401)) := by
  have c6683 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6684 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6685 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6686 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6687 := (leafCircuit_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6688 := (leafCircuit_copies209 a h).1
  have c6689 := (leafCircuit_copies209 a h).2.1
  have c6690 := (leafCircuit_copies209 a h).2.2.1
  have c6691 := (leafCircuit_copies209 a h).2.2.2.1
  have c6692 := (leafCircuit_copies209 a h).2.2.2.2.1
  have c6693 := (leafCircuit_copies209 a h).2.2.2.2.2.1
  have c6694 := (leafCircuit_copies209 a h).2.2.2.2.2.2.1
  have c6695 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.1
  have c6696 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_170_13 := arithEq_of_rows h (row := 170) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_13
  simp only [← c6683, k3, ← c6684, k3, ← c6685] at e_170_13
  have e_170_14 := arithEq_of_rows h (row := 170) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_14
  simp only [← c6692, ← c6693, k3, ← c6694] at e_170_14
  have e_173_0 := arithEq_of_rows h (row := 173) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_0
  simp only [← c6686, ← c6687, ← c6688] at e_173_0
  have e_173_1 := arithEq_of_rows h (row := 173) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_1
  simp only [← c6689, ← c6690, ← c6691] at e_173_1
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_173_0
    linear_combination c6695.trans k0 - hc
  · have hc := e_170_14
    simp only [e_170_13, e_173_1] at hc
    linear_combination c6696.trans k0 - hc

theorem leafCircuit_f1274 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 220)) (a (.virt 292)) (a (.virt 402)) (a (.virt 403)) := by
  have c6697 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.1
  have c6698 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6699 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6700 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6701 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6702 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6703 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6704 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6705 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6706 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6707 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6708 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6709 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6710 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6711 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6712 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6713 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_173_2 := arithEq_of_rows h (row := 173) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_2
  simp only [← c6703, ← c6704, ← c6705] at e_173_2
  have e_173_3 := arithEq_of_rows h (row := 173) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_3
  simp only [← c6706, ← c6707, ← c6708] at e_173_3
  have e_174_0 := arithEq_of_rows h (row := 174) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_0
  simp only [← c6697, k3, ← c6698, k3, ← c6699] at e_174_0
  have e_174_1 := arithEq_of_rows h (row := 174) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_1
  simp only [← c6700, ← c6701, k3, ← c6702, k3] at e_174_1
  have e_174_2 := arithEq_of_rows h (row := 174) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_2
  simp only [← c6709, ← c6710, k3, ← c6711] at e_174_2
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_173_2
    simp only [e_174_1] at hc
    linear_combination c6712.trans k0 - hc
  · have hc := e_174_2
    simp only [e_174_0, e_173_3, e_174_1] at hc
    linear_combination c6713.trans k0 - hc

theorem leafCircuit_f1275 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 220)) (a (.virt 294)) (a (.virt 404)) (a (.virt 405)) := by
  have c6714 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6715 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6716 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6717 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6718 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6719 := (leafCircuit_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6720 := (leafCircuit_copies210 a h).1
  have c6721 := (leafCircuit_copies210 a h).2.1
  have c6722 := (leafCircuit_copies210 a h).2.2.1
  have c6723 := (leafCircuit_copies210 a h).2.2.2.1
  have c6724 := (leafCircuit_copies210 a h).2.2.2.2.1
  have c6725 := (leafCircuit_copies210 a h).2.2.2.2.2.1
  have c6726 := (leafCircuit_copies210 a h).2.2.2.2.2.2.1
  have c6727 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.1
  have c6728 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.1
  have c6729 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.1
  have c6730 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_173_4 := arithEq_of_rows h (row := 173) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_4
  simp only [← c6720, ← c6721, ← c6722] at e_173_4
  have e_173_5 := arithEq_of_rows h (row := 173) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_5
  simp only [← c6723, ← c6724, ← c6725] at e_173_5
  have e_174_3 := arithEq_of_rows h (row := 174) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_3
  simp only [← c6714, k3, ← c6715, k3, ← c6716] at e_174_3
  have e_174_4 := arithEq_of_rows h (row := 174) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_4
  simp only [← c6717, ← c6718, k3, ← c6719, k5] at e_174_4
  have e_174_5 := arithEq_of_rows h (row := 174) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_5
  simp only [← c6726, ← c6727, k3, ← c6728] at e_174_5
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_173_4
    simp only [e_174_4] at hc
    linear_combination c6729.trans k0 - hc
  · have hc := e_174_5
    simp only [e_174_3, e_173_5, e_174_4] at hc
    linear_combination c6730.trans k0 - hc

theorem leafCircuit_f1276 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 220)) (a (.virt 295)) (a (.virt 406)) (a (.virt 407)) := by
  have c6731 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6732 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6733 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6734 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6735 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6736 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6737 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6738 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6739 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6740 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6741 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6742 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6743 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6744 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6745 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6746 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6747 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_173_6 := arithEq_of_rows h (row := 173) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_6
  simp only [← c6737, ← c6738, ← c6739] at e_173_6
  have e_173_7 := arithEq_of_rows h (row := 173) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_7
  simp only [← c6740, ← c6741, ← c6742] at e_173_7
  have e_174_6 := arithEq_of_rows h (row := 174) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_6
  simp only [← c6731, k3, ← c6732, k3, ← c6733] at e_174_6
  have e_174_7 := arithEq_of_rows h (row := 174) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_7
  simp only [← c6734, ← c6735, k3, ← c6736, k6] at e_174_7
  have e_174_8 := arithEq_of_rows h (row := 174) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_8
  simp only [← c6743, ← c6744, k3, ← c6745] at e_174_8
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_173_6
    simp only [e_174_7] at hc
    linear_combination c6746.trans k0 - hc
  · have hc := e_174_8
    simp only [e_174_6, e_173_7, e_174_7] at hc
    linear_combination c6747.trans k0 - hc

theorem leafCircuit_f1277 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 174 43) = bselect (a (.virt 400)) (a (.wire 165 55)) (a (.virt 171)) := by
  have c6748 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6749 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6750 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6751 := (leafCircuit_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6752 := (leafCircuit_copies211 a h).1
  have c6753 := (leafCircuit_copies211 a h).2.1
  have e_174_9 := arithEq_of_rows h (row := 174) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_9
  simp only [← c6748, ← c6749, ← c6750] at e_174_9
  have e_174_10 := arithEq_of_rows h (row := 174) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_10
  simp only [← c6751, ← c6752, ← c6753] at e_174_10
  have hr := e_174_10
  simp only [e_174_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1278 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 174 51) = bselect (a (.virt 400)) (a (.wire 170 3)) (a (.virt 172)) := by
  have c6754 := (leafCircuit_copies211 a h).2.2.1
  have c6755 := (leafCircuit_copies211 a h).2.2.2.1
  have c6756 := (leafCircuit_copies211 a h).2.2.2.2.1
  have c6757 := (leafCircuit_copies211 a h).2.2.2.2.2.1
  have c6758 := (leafCircuit_copies211 a h).2.2.2.2.2.2.1
  have c6759 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.1
  have e_174_11 := arithEq_of_rows h (row := 174) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_11
  simp only [← c6754, ← c6755, ← c6756] at e_174_11
  have e_174_12 := arithEq_of_rows h (row := 174) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_12
  simp only [← c6757, ← c6758, ← c6759] at e_174_12
  have hr := e_174_12
  simp only [e_174_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1279 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 174 59) = bselect (a (.virt 400)) (a (.wire 170 11)) (a (.virt 173)) := by
  have c6760 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.1
  have c6761 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.1
  have c6762 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6763 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6764 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6765 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_174_13 := arithEq_of_rows h (row := 174) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_13
  simp only [← c6760, ← c6761, ← c6762] at e_174_13
  have e_174_14 := arithEq_of_rows h (row := 174) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_14
  simp only [← c6763, ← c6764, ← c6765] at e_174_14
  have hr := e_174_14
  simp only [e_174_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1280 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 7) = bselect (a (.virt 400)) (a (.wire 170 19)) (a (.virt 174)) := by
  have c6766 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6767 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6768 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6769 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6770 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6771 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_0 := arithEq_of_rows h (row := 175) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_0
  simp only [← c6766, ← c6767, ← c6768] at e_175_0
  have e_175_1 := arithEq_of_rows h (row := 175) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_1
  simp only [← c6769, ← c6770, ← c6771] at e_175_1
  have hr := e_175_1
  simp only [e_175_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1281 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 15) = bselect (a (.virt 400)) (a (.virt 171)) (a (.virt 175)) := by
  have c6772 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6773 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6774 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6775 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6776 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6777 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_2 := arithEq_of_rows h (row := 175) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_2
  simp only [← c6772, ← c6773, ← c6774] at e_175_2
  have e_175_3 := arithEq_of_rows h (row := 175) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_3
  simp only [← c6775, ← c6776, ← c6777] at e_175_3
  have hr := e_175_3
  simp only [e_175_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1282 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 23) = bselect (a (.virt 402)) (a (.wire 165 55)) (a (.wire 175 15)) := by
  have c6778 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6779 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6780 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6781 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6782 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6783 := (leafCircuit_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_175_4 := arithEq_of_rows h (row := 175) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_4
  simp only [← c6778, ← c6779, ← c6780] at e_175_4
  have e_175_5 := arithEq_of_rows h (row := 175) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_5
  simp only [← c6781, ← c6782, ← c6783] at e_175_5
  have hr := e_175_5
  simp only [e_175_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1283 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 31) = bselect (a (.virt 400)) (a (.virt 172)) (a (.virt 176)) := by
  have c6784 := (leafCircuit_copies212 a h).1
  have c6785 := (leafCircuit_copies212 a h).2.1
  have c6786 := (leafCircuit_copies212 a h).2.2.1
  have c6787 := (leafCircuit_copies212 a h).2.2.2.1
  have c6788 := (leafCircuit_copies212 a h).2.2.2.2.1
  have c6789 := (leafCircuit_copies212 a h).2.2.2.2.2.1
  have e_175_6 := arithEq_of_rows h (row := 175) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_6
  simp only [← c6784, ← c6785, ← c6786] at e_175_6
  have e_175_7 := arithEq_of_rows h (row := 175) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_7
  simp only [← c6787, ← c6788, ← c6789] at e_175_7
  have hr := e_175_7
  simp only [e_175_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1284 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 39) = bselect (a (.virt 402)) (a (.wire 170 3)) (a (.wire 175 31)) := by
  have c6790 := (leafCircuit_copies212 a h).2.2.2.2.2.2.1
  have c6791 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.1
  have c6792 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.1
  have c6793 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.1
  have c6794 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6795 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_8 := arithEq_of_rows h (row := 175) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_8
  simp only [← c6790, ← c6791, ← c6792] at e_175_8
  have e_175_9 := arithEq_of_rows h (row := 175) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_9
  simp only [← c6793, ← c6794, ← c6795] at e_175_9
  have hr := e_175_9
  simp only [e_175_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1285 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 47) = bselect (a (.virt 400)) (a (.virt 173)) (a (.virt 177)) := by
  have c6796 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6797 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6798 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6799 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6800 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6801 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_10 := arithEq_of_rows h (row := 175) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_10
  simp only [← c6796, ← c6797, ← c6798] at e_175_10
  have e_175_11 := arithEq_of_rows h (row := 175) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_11
  simp only [← c6799, ← c6800, ← c6801] at e_175_11
  have hr := e_175_11
  simp only [e_175_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1286 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 175 55) = bselect (a (.virt 402)) (a (.wire 170 11)) (a (.wire 175 47)) := by
  have c6802 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6803 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6804 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6805 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6806 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6807 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_12 := arithEq_of_rows h (row := 175) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_12
  simp only [← c6802, ← c6803, ← c6804] at e_175_12
  have e_175_13 := arithEq_of_rows h (row := 175) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_13
  simp only [← c6805, ← c6806, ← c6807] at e_175_13
  have hr := e_175_13
  simp only [e_175_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1287 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 3) = bselect (a (.virt 400)) (a (.virt 174)) (a (.virt 178)) := by
  have c6808 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6809 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6810 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6811 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6812 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6813 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_14 := arithEq_of_rows h (row := 175) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_14
  simp only [← c6808, ← c6809, ← c6810] at e_175_14
  have e_176_0 := arithEq_of_rows h (row := 176) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_0
  simp only [← c6811, ← c6812, ← c6813] at e_176_0
  have hr := e_176_0
  simp only [e_175_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1288 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 11) = bselect (a (.virt 402)) (a (.wire 170 19)) (a (.wire 176 3)) := by
  have c6814 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6815 := (leafCircuit_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6816 := (leafCircuit_copies213 a h).1
  have c6817 := (leafCircuit_copies213 a h).2.1
  have c6818 := (leafCircuit_copies213 a h).2.2.1
  have c6819 := (leafCircuit_copies213 a h).2.2.2.1
  have e_176_1 := arithEq_of_rows h (row := 176) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_1
  simp only [← c6814, ← c6815, ← c6816] at e_176_1
  have e_176_2 := arithEq_of_rows h (row := 176) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_2
  simp only [← c6817, ← c6818, ← c6819] at e_176_2
  have hr := e_176_2
  simp only [e_176_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1289 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) := by
  have c6820 := (leafCircuit_copies213 a h).2.2.2.2.1
  have c6821 := (leafCircuit_copies213 a h).2.2.2.2.2.1
  have c6822 := (leafCircuit_copies213 a h).2.2.2.2.2.2.1
  have c6823 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.1
  have c6824 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.1
  have c6825 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_2 := arithEq_of_rows h (row := 164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_2
  simp only [← c6820, ← c6821, ← c6822] at e_164_2
  have e_167_11 := arithEq_of_rows h (row := 167) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_11
  simp only [← c6823, ← c6824, k3, ← c6825] at e_167_11
  have hr := e_167_11
  simp only [e_164_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1290 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 19) = bselect (a (.wire 167 47)) (a (.virt 175)) (a (.virt 179)) := by
  have c6826 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6827 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6828 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6829 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6830 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6831 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_176_3 := arithEq_of_rows h (row := 176) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_3
  simp only [← c6826, ← c6827, ← c6828] at e_176_3
  have e_176_4 := arithEq_of_rows h (row := 176) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_4
  simp only [← c6829, ← c6830, ← c6831] at e_176_4
  have hr := e_176_4
  simp only [e_176_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1291 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 27) = bselect (a (.virt 404)) (a (.wire 165 55)) (a (.wire 176 19)) := by
  have c6832 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6833 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6834 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6835 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6836 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6837 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_176_5 := arithEq_of_rows h (row := 176) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_5
  simp only [← c6832, ← c6833, ← c6834] at e_176_5
  have e_176_6 := arithEq_of_rows h (row := 176) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_6
  simp only [← c6835, ← c6836, ← c6837] at e_176_6
  have hr := e_176_6
  simp only [e_176_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1292 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) := by
  have c6820 := (leafCircuit_copies213 a h).2.2.2.2.1
  have c6821 := (leafCircuit_copies213 a h).2.2.2.2.2.1
  have c6822 := (leafCircuit_copies213 a h).2.2.2.2.2.2.1
  have c6823 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.1
  have c6824 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.1
  have c6825 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_2 := arithEq_of_rows h (row := 164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_2
  simp only [← c6820, ← c6821, ← c6822] at e_164_2
  have e_167_11 := arithEq_of_rows h (row := 167) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_11
  simp only [← c6823, ← c6824, k3, ← c6825] at e_167_11
  have hr := e_167_11
  simp only [e_164_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1293 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 35) = bselect (a (.wire 167 47)) (a (.virt 176)) (a (.virt 180)) := by
  have c6838 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6839 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6840 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6841 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6842 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6843 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_176_7 := arithEq_of_rows h (row := 176) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_7
  simp only [← c6838, ← c6839, ← c6840] at e_176_7
  have e_176_8 := arithEq_of_rows h (row := 176) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_8
  simp only [← c6841, ← c6842, ← c6843] at e_176_8
  have hr := e_176_8
  simp only [e_176_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1294 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 43) = bselect (a (.virt 404)) (a (.wire 170 3)) (a (.wire 176 35)) := by
  have c6844 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6845 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6846 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6847 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6848 := (leafCircuit_copies214 a h).1
  have c6849 := (leafCircuit_copies214 a h).2.1
  have e_176_9 := arithEq_of_rows h (row := 176) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_9
  simp only [← c6844, ← c6845, ← c6846] at e_176_9
  have e_176_10 := arithEq_of_rows h (row := 176) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_10
  simp only [← c6847, ← c6848, ← c6849] at e_176_10
  have hr := e_176_10
  simp only [e_176_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1295 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) := by
  have c6820 := (leafCircuit_copies213 a h).2.2.2.2.1
  have c6821 := (leafCircuit_copies213 a h).2.2.2.2.2.1
  have c6822 := (leafCircuit_copies213 a h).2.2.2.2.2.2.1
  have c6823 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.1
  have c6824 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.1
  have c6825 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_2 := arithEq_of_rows h (row := 164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_2
  simp only [← c6820, ← c6821, ← c6822] at e_164_2
  have e_167_11 := arithEq_of_rows h (row := 167) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_11
  simp only [← c6823, ← c6824, k3, ← c6825] at e_167_11
  have hr := e_167_11
  simp only [e_164_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1296 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 51) = bselect (a (.wire 167 47)) (a (.virt 177)) (a (.virt 181)) := by
  have c6850 := (leafCircuit_copies214 a h).2.2.1
  have c6851 := (leafCircuit_copies214 a h).2.2.2.1
  have c6852 := (leafCircuit_copies214 a h).2.2.2.2.1
  have c6853 := (leafCircuit_copies214 a h).2.2.2.2.2.1
  have c6854 := (leafCircuit_copies214 a h).2.2.2.2.2.2.1
  have c6855 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.1
  have e_176_11 := arithEq_of_rows h (row := 176) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_11
  simp only [← c6850, ← c6851, ← c6852] at e_176_11
  have e_176_12 := arithEq_of_rows h (row := 176) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_12
  simp only [← c6853, ← c6854, ← c6855] at e_176_12
  have hr := e_176_12
  simp only [e_176_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1297 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 176 59) = bselect (a (.virt 404)) (a (.wire 170 11)) (a (.wire 176 51)) := by
  have c6856 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.1
  have c6857 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.1
  have c6858 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6859 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6860 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6861 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_176_13 := arithEq_of_rows h (row := 176) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_13
  simp only [← c6856, ← c6857, ← c6858] at e_176_13
  have e_176_14 := arithEq_of_rows h (row := 176) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_14
  simp only [← c6859, ← c6860, ← c6861] at e_176_14
  have hr := e_176_14
  simp only [e_176_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1298 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) := by
  have c6820 := (leafCircuit_copies213 a h).2.2.2.2.1
  have c6821 := (leafCircuit_copies213 a h).2.2.2.2.2.1
  have c6822 := (leafCircuit_copies213 a h).2.2.2.2.2.2.1
  have c6823 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.1
  have c6824 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.1
  have c6825 := (leafCircuit_copies213 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_2 := arithEq_of_rows h (row := 164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_2
  simp only [← c6820, ← c6821, ← c6822] at e_164_2
  have e_167_11 := arithEq_of_rows h (row := 167) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_11
  simp only [← c6823, ← c6824, k3, ← c6825] at e_167_11
  have hr := e_167_11
  simp only [e_164_2] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1299 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 7) = bselect (a (.wire 167 47)) (a (.virt 178)) (a (.virt 182)) := by
  have c6862 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6863 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6864 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6865 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6866 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6867 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_177_0 := arithEq_of_rows h (row := 177) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_0
  simp only [← c6862, ← c6863, ← c6864] at e_177_0
  have e_177_1 := arithEq_of_rows h (row := 177) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_1
  simp only [← c6865, ← c6866, ← c6867] at e_177_1
  have hr := e_177_1
  simp only [e_177_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1300 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 15) = bselect (a (.virt 404)) (a (.wire 170 19)) (a (.wire 177 7)) := by
  have c6868 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6869 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6870 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6871 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6872 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6873 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_177_2 := arithEq_of_rows h (row := 177) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_2
  simp only [← c6868, ← c6869, ← c6870] at e_177_2
  have e_177_3 := arithEq_of_rows h (row := 177) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_3
  simp only [← c6871, ← c6872, ← c6873] at e_177_3
  have hr := e_177_3
  simp only [e_177_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1301 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 23) = bselect (a (.virt 406)) (a (.wire 165 55)) (a (.virt 179)) := by
  have c6874 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6875 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6876 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6877 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6878 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6879 := (leafCircuit_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_177_4 := arithEq_of_rows h (row := 177) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_4
  simp only [← c6874, ← c6875, ← c6876] at e_177_4
  have e_177_5 := arithEq_of_rows h (row := 177) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_5
  simp only [← c6877, ← c6878, ← c6879] at e_177_5
  have hr := e_177_5
  simp only [e_177_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1302 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 31) = bselect (a (.virt 406)) (a (.wire 170 3)) (a (.virt 180)) := by
  have c6880 := (leafCircuit_copies215 a h).1
  have c6881 := (leafCircuit_copies215 a h).2.1
  have c6882 := (leafCircuit_copies215 a h).2.2.1
  have c6883 := (leafCircuit_copies215 a h).2.2.2.1
  have c6884 := (leafCircuit_copies215 a h).2.2.2.2.1
  have c6885 := (leafCircuit_copies215 a h).2.2.2.2.2.1
  have e_177_6 := arithEq_of_rows h (row := 177) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_6
  simp only [← c6880, ← c6881, ← c6882] at e_177_6
  have e_177_7 := arithEq_of_rows h (row := 177) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_7
  simp only [← c6883, ← c6884, ← c6885] at e_177_7
  have hr := e_177_7
  simp only [e_177_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1303 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 39) = bselect (a (.virt 406)) (a (.wire 170 11)) (a (.virt 181)) := by
  have c6886 := (leafCircuit_copies215 a h).2.2.2.2.2.2.1
  have c6887 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.1
  have c6888 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.1
  have c6889 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.1
  have c6890 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6891 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_177_8 := arithEq_of_rows h (row := 177) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_8
  simp only [← c6886, ← c6887, ← c6888] at e_177_8
  have e_177_9 := arithEq_of_rows h (row := 177) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_9
  simp only [← c6889, ← c6890, ← c6891] at e_177_9
  have hr := e_177_9
  simp only [e_177_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1304 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 47) = bselect (a (.virt 406)) (a (.wire 170 19)) (a (.virt 182)) := by
  have c6892 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6893 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6894 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6895 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6896 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6897 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_177_10 := arithEq_of_rows h (row := 177) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_10
  simp only [← c6892, ← c6893, ← c6894] at e_177_10
  have e_177_11 := arithEq_of_rows h (row := 177) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_11
  simp only [← c6895, ← c6896, ← c6897] at e_177_11
  have hr := e_177_11
  simp only [e_177_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1305 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 181 12) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 0 ∧ a (.wire 181 13) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 1 ∧ a (.wire 181 14) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 2 ∧ a (.wire 181 15) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 3) := by
  have c6898 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6899 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6900 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6901 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6902 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6903 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6904 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6905 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6906 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6907 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6908 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6909 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6910 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6911 := (leafCircuit_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6912 := (leafCircuit_copies216 a h).1
  have c6913 := (leafCircuit_copies216 a h).2.1
  have c6914 := (leafCircuit_copies216 a h).2.2.1
  have c6915 := (leafCircuit_copies216 a h).2.2.2.1
  have c6916 := (leafCircuit_copies216 a h).2.2.2.2.1
  have c6917 := (leafCircuit_copies216 a h).2.2.2.2.2.1
  have c6918 := (leafCircuit_copies216 a h).2.2.2.2.2.2.1
  have c6919 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.1
  have c6920 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.1
  have c6921 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.1
  have c6922 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6923 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6924 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6925 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6926 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6927 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6928 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6929 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6930 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6931 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6932 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6933 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6934 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6935 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6936 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6937 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6938 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6939 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6940 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6941 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6942 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6943 := (leafCircuit_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6944 := (leafCircuit_copies217 a h).1
  have c6945 := (leafCircuit_copies217 a h).2.1
  have c6946 := (leafCircuit_copies217 a h).2.2.1
  have c6947 := (leafCircuit_copies217 a h).2.2.2.1
  have c6948 := (leafCircuit_copies217 a h).2.2.2.2.1
  have c6949 := (leafCircuit_copies217 a h).2.2.2.2.2.1
  have c6950 := (leafCircuit_copies217 a h).2.2.2.2.2.2.1
  have c6951 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.1
  have c6952 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.1
  have c6953 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.1
  have c6954 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6955 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6956 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6957 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6958 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6959 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6960 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_167_12 := arithEq_of_rows h (row := 167) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_12
  simp only [← c6910, ← c6911, k3, ← c6912] at e_167_12
  have e_167_13 := arithEq_of_rows h (row := 167) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_13
  simp only [← c6913, ← c6914, k3, ← c6915] at e_167_13
  have e_167_14 := arithEq_of_rows h (row := 167) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_14
  simp only [← c6916, ← c6917, k3, ← c6918] at e_167_14
  have e_179_0 := arithEq_of_rows h (row := 179) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_0
  simp only [← c6919, ← c6920, k3, ← c6921] at e_179_0
  have e_179_1 := arithEq_of_rows h (row := 179) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_1
  simp only [← c6922, ← c6923, k3, ← c6924] at e_179_1
  have e_179_2 := arithEq_of_rows h (row := 179) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_2
  simp only [← c6925, ← c6926, k3, ← c6927] at e_179_2
  have e_179_3 := arithEq_of_rows h (row := 179) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_3
  simp only [← c6928, ← c6929, k3, ← c6930] at e_179_3
  have e_179_4 := arithEq_of_rows h (row := 179) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_4
  simp only [← c6931, ← c6932, k3, ← c6933] at e_179_4
  have e_179_5 := arithEq_of_rows h (row := 179) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_5
  simp only [← c6946, ← c6947, k3, ← c6948, k3] at e_179_5
  have hin0 : poseidon2In a 178 = addBlock (fun _ => 0) [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11)] :=
    poseidon2In_first c6898.symm c6899.symm c6900.symm c6901.symm c6902.symm c6903.symm c6904.symm c6905.symm (c6906.symm.trans k0) (c6907.symm.trans k0) (c6908.symm.trans k0) (c6909.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 178) rfl rfl hin0
  have hin1 : poseidon2In a 180 = addBlock (poseidon2Out a 178) [a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] :=
    poseidon2In_chain (by linear_combination -c6934 + e_167_12) (by linear_combination -c6935 + e_167_13) (by linear_combination -c6936 + e_167_14) (by linear_combination -c6937 + e_179_0) (by linear_combination -c6938 + e_179_1) (by linear_combination -c6939 + e_179_2) (by linear_combination -c6940 + e_179_3) (by linear_combination -c6941 + e_179_4) c6942.symm c6943.symm c6944.symm c6945.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 180) rfl rfl hin1
  have hin2 : poseidon2In a 181 = addBlock (poseidon2Out a 180) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c6949 + e_179_5) (c6950.symm.trans (add_zero _).symm) (c6951.symm.trans (add_zero _).symm) (c6952.symm.trans (add_zero _).symm) (c6953.symm.trans (add_zero _).symm) (c6954.symm.trans (add_zero _).symm) (c6955.symm.trans (add_zero _).symm) (c6956.symm.trans (add_zero _).symm) c6957.symm c6958.symm c6959.symm c6960.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 181) rfl rfl hin2
  have hpad : pad10 [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] =
      [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11)] ++ ([a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1306 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 177 55) = bselect (a (.wire 167 39)) (a (.wire 181 12)) (a (.wire 165 55)) := by
  have c6961 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6962 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6963 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6964 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6965 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6966 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_177_12 := arithEq_of_rows h (row := 177) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_12
  simp only [← c6961, ← c6962, ← c6963] at e_177_12
  have e_177_13 := arithEq_of_rows h (row := 177) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_13
  simp only [← c6964, ← c6965, ← c6966] at e_177_13
  have hr := e_177_13
  simp only [e_177_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1307 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 3) = bselect (a (.wire 167 39)) (a (.wire 181 13)) (a (.wire 170 3)) := by
  have c6967 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6968 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6969 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6970 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6971 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6972 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_177_14 := arithEq_of_rows h (row := 177) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_14
  simp only [← c6967, ← c6968, ← c6969] at e_177_14
  have e_182_0 := arithEq_of_rows h (row := 182) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_0
  simp only [← c6970, ← c6971, ← c6972] at e_182_0
  have hr := e_182_0
  simp only [e_177_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1308 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 11) = bselect (a (.wire 167 39)) (a (.wire 181 14)) (a (.wire 170 11)) := by
  have c6973 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6974 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6975 := (leafCircuit_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6976 := (leafCircuit_copies218 a h).1
  have c6977 := (leafCircuit_copies218 a h).2.1
  have c6978 := (leafCircuit_copies218 a h).2.2.1
  have e_182_1 := arithEq_of_rows h (row := 182) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_1
  simp only [← c6973, ← c6974, ← c6975] at e_182_1
  have e_182_2 := arithEq_of_rows h (row := 182) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_2
  simp only [← c6976, ← c6977, ← c6978] at e_182_2
  have hr := e_182_2
  simp only [e_182_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1309 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 19) = bselect (a (.wire 167 39)) (a (.wire 181 15)) (a (.wire 170 19)) := by
  have c6979 := (leafCircuit_copies218 a h).2.2.2.1
  have c6980 := (leafCircuit_copies218 a h).2.2.2.2.1
  have c6981 := (leafCircuit_copies218 a h).2.2.2.2.2.1
  have c6982 := (leafCircuit_copies218 a h).2.2.2.2.2.2.1
  have c6983 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.1
  have c6984 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.1
  have e_182_3 := arithEq_of_rows h (row := 182) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_3
  simp only [← c6979, ← c6980, ← c6981] at e_182_3
  have e_182_4 := arithEq_of_rows h (row := 182) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_4
  simp only [← c6982, ← c6983, ← c6984] at e_182_4
  have hr := e_182_4
  simp only [e_182_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1310 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 183 1), a (.wire 183 2), a (.wire 183 3), a (.wire 183 4), a (.wire 183 5)] := by
  have c6985 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.1
  have c6986 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6987 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6988 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6989 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6990 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6991 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6992 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6993 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6994 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6995 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6996 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6997 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6998 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6999 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7000 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7001 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7002 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7003 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7004 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7005 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7006 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7007 := (leafCircuit_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7008 := (leafCircuit_copies219 a h).1
  have c7009 := (leafCircuit_copies219 a h).2.1
  have c7010 := (leafCircuit_copies219 a h).2.2.1
  have c7011 := (leafCircuit_copies219 a h).2.2.2.1
  have c7012 := (leafCircuit_copies219 a h).2.2.2.2.1
  have c7013 := (leafCircuit_copies219 a h).2.2.2.2.2.1
  have c7014 := (leafCircuit_copies219 a h).2.2.2.2.2.2.1
  have c7015 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.1
  have c7016 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.1
  have c7017 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.1
  have c7018 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7019 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7020 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7021 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7022 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7023 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7024 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7025 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7026 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7027 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7028 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7029 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7030 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7031 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7032 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7033 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7034 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7035 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7036 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7037 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7038 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7039 := (leafCircuit_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 183) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c6985.trans k0
    · exact c6986.trans k0
    · exact c6987.trans k0
    · exact c6988.trans k0
    · exact c6989.trans k0
    · exact c6990.trans k0
    · exact c6991.trans k0
    · exact c6992.trans k0
    · exact c6993.trans k0
    · exact c6994.trans k0
    · exact c6995.trans k0
    · exact c6996.trans k0
    · exact c6997.trans k0
    · exact c6998.trans k0
    · exact c6999.trans k0
    · exact c7000.trans k0
    · exact c7001.trans k0
    · exact c7002.trans k0
    · exact c7003.trans k0
    · exact c7004.trans k0
    · exact c7005.trans k0
    · exact c7006.trans k0
    · exact c7007.trans k0
    · exact c7008.trans k0
    · exact c7009.trans k0
    · exact c7010.trans k0
    · exact c7011.trans k0
    · exact c7012.trans k0
    · exact c7013.trans k0
    · exact c7014.trans k0
    · exact c7015.trans k0
    · exact c7016.trans k0
    · exact c7017.trans k0
    · exact c7018.trans k0
    · exact c7019.trans k0
    · exact c7020.trans k0
    · exact c7021.trans k0
    · exact c7022.trans k0
    · exact c7023.trans k0
    · exact c7024.trans k0
    · exact c7025.trans k0
    · exact c7026.trans k0
    · exact c7027.trans k0
    · exact c7028.trans k0
    · exact c7029.trans k0
    · exact c7030.trans k0
    · exact c7031.trans k0
    · exact c7032.trans k0
    · exact c7033.trans k0
    · exact c7034.trans k0
    · exact c7035.trans k0
    · exact c7036.trans k0
    · exact c7037.trans k0
    · exact c7038.trans k0
    )
  rwa [c7039] at hr

theorem leafCircuit_f1311 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1312 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = band (a (.virt 292)) (a (.wire 183 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1313 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = band (a (.wire 183 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1314 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = bor (a (.virt 14)) (a (.wire 183 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1315 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 183 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1316 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1317 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = a (.virt 14) + a (.wire 183 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1318 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = a (.wire 183 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1319 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 23) = bnot (a (.wire 183 5)) := by
  have c7040 := (leafCircuit_copies220 a h).1
  have c7041 := (leafCircuit_copies220 a h).2.1
  have c7042 := (leafCircuit_copies220 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_5 := arithEq_of_rows h (row := 182) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_5
  simp only [← c7040, k3, ← c7041, k3, ← c7042] at e_182_5
  have hr := e_182_5
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1320 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 23) = band (a (.virt 292)) (a (.wire 182 23)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1321 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1322 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 183 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1323 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 182 23)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1324 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = bor (a (.wire 183 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1325 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 4) = a (.virt 292) * a (.wire 183 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1326 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 35) = a (.virt 294) * a (.wire 183 4) := by
  have c7043 := (leafCircuit_copies220 a h).2.2.2.1
  have c7044 := (leafCircuit_copies220 a h).2.2.2.2.1
  have c7045 := (leafCircuit_copies220 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_173_8 := arithEq_of_rows h (row := 173) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_8
  simp only [← c7043, k5, ← c7044, ← c7045, k5] at e_173_8
  have hr := e_173_8
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1327 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 27) = a (.virt 292) + a (.wire 183 4) := by
  have c7046 := (leafCircuit_copies220 a h).2.2.2.2.2.2.1
  have c7047 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.1
  have c7048 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_179_6 := arithEq_of_rows h (row := 179) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_6
  simp only [← c7046, k3, ← c7047, k3, ← c7048] at e_179_6
  have hr := e_179_6
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1328 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 27) = a (.wire 179 27) - a (.wire 173 35) := by
  have c7049 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.1
  have c7050 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7051 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_6 := arithEq_of_rows h (row := 182) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_6
  simp only [← c7049, ← c7050, k3, ← c7051] at e_182_6
  have hr := e_182_6
  linear_combination hr

theorem leafCircuit_f1329 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 31) = bnot (a (.wire 182 27)) := by
  have c7052 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7053 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7054 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_7 := arithEq_of_rows h (row := 182) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_7
  simp only [← c7052, k3, ← c7053, k3, ← c7054] at e_182_7
  have hr := e_182_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1330 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 39) = band (a (.wire 182 23)) (a (.wire 182 31)) := by
  have c7055 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7056 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7057 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_9 := arithEq_of_rows h (row := 173) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_9
  simp only [← c7055, ← c7056, ← c7057] at e_173_9
  have hr := e_173_9
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1331 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1332 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 183 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1333 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 173 39)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1334 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = bor (a (.wire 183 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1335 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 3) = a (.virt 292) * a (.wire 183 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1336 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 43) = a (.virt 294) * a (.wire 183 3) := by
  have c7058 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7059 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7060 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_173_10 := arithEq_of_rows h (row := 173) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_10
  simp only [← c7058, k5, ← c7059, ← c7060, k5] at e_173_10
  have hr := e_173_10
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1337 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 31) = a (.virt 292) + a (.wire 183 3) := by
  have c7061 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7062 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7063 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_179_7 := arithEq_of_rows h (row := 179) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_7
  simp only [← c7061, k3, ← c7062, k3, ← c7063] at e_179_7
  have hr := e_179_7
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1338 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 35) = a (.wire 179 31) - a (.wire 173 43) := by
  have c7064 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7065 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7066 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_8 := arithEq_of_rows h (row := 182) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_8
  simp only [← c7064, ← c7065, k3, ← c7066] at e_182_8
  have hr := e_182_8
  linear_combination hr

theorem leafCircuit_f1339 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 39) = bnot (a (.wire 182 35)) := by
  have c7067 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7068 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7069 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_9 := arithEq_of_rows h (row := 182) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_9
  simp only [← c7067, k3, ← c7068, k3, ← c7069] at e_182_9
  have hr := e_182_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1340 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 47) = band (a (.wire 173 39)) (a (.wire 182 39)) := by
  have c7070 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7071 := (leafCircuit_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7072 := (leafCircuit_copies221 a h).1
  have e_173_11 := arithEq_of_rows h (row := 173) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_11
  simp only [← c7070, ← c7071, ← c7072] at e_173_11
  have hr := e_173_11
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1341 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1342 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 183 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1343 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 173 47)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1344 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 5) = bor (a (.wire 183 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1345 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 2) = a (.virt 292) * a (.wire 183 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1346 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 51) = a (.virt 294) * a (.wire 183 2) := by
  have c7073 := (leafCircuit_copies221 a h).2.1
  have c7074 := (leafCircuit_copies221 a h).2.2.1
  have c7075 := (leafCircuit_copies221 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_173_12 := arithEq_of_rows h (row := 173) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_12
  simp only [← c7073, k5, ← c7074, ← c7075, k5] at e_173_12
  have hr := e_173_12
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1347 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 35) = a (.virt 292) + a (.wire 183 2) := by
  have c7076 := (leafCircuit_copies221 a h).2.2.2.2.1
  have c7077 := (leafCircuit_copies221 a h).2.2.2.2.2.1
  have c7078 := (leafCircuit_copies221 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_179_8 := arithEq_of_rows h (row := 179) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_8
  simp only [← c7076, k3, ← c7077, k3, ← c7078] at e_179_8
  have hr := e_179_8
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1348 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 43) = a (.wire 179 35) - a (.wire 173 51) := by
  have c7079 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.1
  have c7080 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.1
  have c7081 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_10 := arithEq_of_rows h (row := 182) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_10
  simp only [← c7079, ← c7080, k3, ← c7081] at e_182_10
  have hr := e_182_10
  linear_combination hr

theorem leafCircuit_f1349 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 47) = bnot (a (.wire 182 43)) := by
  have c7082 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7083 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7084 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_11 := arithEq_of_rows h (row := 182) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_11
  simp only [← c7082, k3, ← c7083, k3, ← c7084] at e_182_11
  have hr := e_182_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1350 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 55) = band (a (.wire 173 47)) (a (.wire 182 47)) := by
  have c7085 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7086 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7087 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_13 := arithEq_of_rows h (row := 173) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_13
  simp only [← c7085, ← c7086, ← c7087] at e_173_13
  have hr := e_173_13
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1351 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1352 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 1) = band (a (.virt 292)) (a (.wire 183 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1353 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 173 59) = band (a (.wire 183 1)) (a (.wire 173 55)) := by
  have c7088 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7089 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7090 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_14 := arithEq_of_rows h (row := 173) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_14
  simp only [← c7088, ← c7089, ← c7090] at e_173_14
  have hr := e_173_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1354 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 39) = bor (a (.wire 183 5)) (a (.wire 173 59)) := by
  have c7091 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7092 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7093 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7094 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7095 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7096 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_3 := arithEq_of_rows h (row := 164) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_3
  simp only [← c7091, ← c7092, ← c7093] at e_164_3
  have e_179_9 := arithEq_of_rows h (row := 179) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_9
  simp only [← c7094, ← c7095, k3, ← c7096] at e_179_9
  have hr := e_179_9
  simp only [e_164_3] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1355 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 183 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1356 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1357 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 1) = a (.virt 14) + a (.wire 183 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1358 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 183 1) = a (.wire 183 1) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1359 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 182 51) = bnot (a (.wire 183 1)) := by
  have c7097 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7098 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7099 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_12 := arithEq_of_rows h (row := 182) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_12
  simp only [← c7097, k3, ← c7098, k3, ← c7099] at e_182_12
  have hr := e_182_12
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1360 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 3) = band (a (.wire 173 55)) (a (.wire 182 51)) := by
  have c7100 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7101 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7102 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_184_0 := arithEq_of_rows h (row := 184) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_0
  simp only [← c7100, ← c7101, ← c7102] at e_184_0
  have hr := e_184_0
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1361 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 221)) 2 := by
  have c7103 := (leafCircuit_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7104 := (leafCircuit_copies222 a h).1
  have c7105 := (leafCircuit_copies222 a h).2.1
  have c7106 := (leafCircuit_copies222 a h).2.2.1
  have c7107 := (leafCircuit_copies222 a h).2.2.2.1
  have c7108 := (leafCircuit_copies222 a h).2.2.2.2.1
  have c7109 := (leafCircuit_copies222 a h).2.2.2.2.2.1
  have c7110 := (leafCircuit_copies222 a h).2.2.2.2.2.2.1
  have c7111 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.1
  have c7112 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.1
  have c7113 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.1
  have c7114 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7115 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7116 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7117 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7118 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7119 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7120 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7121 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7122 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7123 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7124 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7125 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7126 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7127 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7128 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7129 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7130 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7131 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7132 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7133 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7134 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7135 := (leafCircuit_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7136 := (leafCircuit_copies223 a h).1
  have c7137 := (leafCircuit_copies223 a h).2.1
  have c7138 := (leafCircuit_copies223 a h).2.2.1
  have c7139 := (leafCircuit_copies223 a h).2.2.2.1
  have c7140 := (leafCircuit_copies223 a h).2.2.2.2.1
  have c7141 := (leafCircuit_copies223 a h).2.2.2.2.2.1
  have c7142 := (leafCircuit_copies223 a h).2.2.2.2.2.2.1
  have c7143 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.1
  have c7144 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.1
  have c7145 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.1
  have c7146 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7147 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7148 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7149 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7150 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7151 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7152 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7153 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7154 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7155 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7156 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7157 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7158 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7159 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7160 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 185) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c7103.trans k0
    · exact c7104.trans k0
    · exact c7105.trans k0
    · exact c7106.trans k0
    · exact c7107.trans k0
    · exact c7108.trans k0
    · exact c7109.trans k0
    · exact c7110.trans k0
    · exact c7111.trans k0
    · exact c7112.trans k0
    · exact c7113.trans k0
    · exact c7114.trans k0
    · exact c7115.trans k0
    · exact c7116.trans k0
    · exact c7117.trans k0
    · exact c7118.trans k0
    · exact c7119.trans k0
    · exact c7120.trans k0
    · exact c7121.trans k0
    · exact c7122.trans k0
    · exact c7123.trans k0
    · exact c7124.trans k0
    · exact c7125.trans k0
    · exact c7126.trans k0
    · exact c7127.trans k0
    · exact c7128.trans k0
    · exact c7129.trans k0
    · exact c7130.trans k0
    · exact c7131.trans k0
    · exact c7132.trans k0
    · exact c7133.trans k0
    · exact c7134.trans k0
    · exact c7135.trans k0
    · exact c7136.trans k0
    · exact c7137.trans k0
    · exact c7138.trans k0
    · exact c7139.trans k0
    · exact c7140.trans k0
    · exact c7141.trans k0
    · exact c7142.trans k0
    · exact c7143.trans k0
    · exact c7144.trans k0
    · exact c7145.trans k0
    · exact c7146.trans k0
    · exact c7147.trans k0
    · exact c7148.trans k0
    · exact c7149.trans k0
    · exact c7150.trans k0
    · exact c7151.trans k0
    · exact c7152.trans k0
    · exact c7153.trans k0
    · exact c7154.trans k0
    · exact c7155.trans k0
    · exact c7156.trans k0
    · exact c7157.trans k0
    · exact c7158.trans k0
    · exact c7159.trans k0
    )
  rwa [c7160] at hr

theorem leafCircuit_f1362 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 221)) (a (.virt 14)) (a (.virt 408)) (a (.virt 409)) := by
  have c7161 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7162 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7163 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7164 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7165 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7166 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7167 := (leafCircuit_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7168 := (leafCircuit_copies224 a h).1
  have c7169 := (leafCircuit_copies224 a h).2.1
  have c7170 := (leafCircuit_copies224 a h).2.2.1
  have c7171 := (leafCircuit_copies224 a h).2.2.2.1
  have c7172 := (leafCircuit_copies224 a h).2.2.2.2.1
  have c7173 := (leafCircuit_copies224 a h).2.2.2.2.2.1
  have c7174 := (leafCircuit_copies224 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_182_13 := arithEq_of_rows h (row := 182) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_13
  simp only [← c7161, k3, ← c7162, k3, ← c7163] at e_182_13
  have e_182_14 := arithEq_of_rows h (row := 182) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_14
  simp only [← c7170, ← c7171, k3, ← c7172] at e_182_14
  have e_184_1 := arithEq_of_rows h (row := 184) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_1
  simp only [← c7164, ← c7165, ← c7166] at e_184_1
  have e_184_2 := arithEq_of_rows h (row := 184) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_2
  simp only [← c7167, ← c7168, ← c7169] at e_184_2
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_184_1
    linear_combination c7173.trans k0 - hc
  · have hc := e_182_14
    simp only [e_182_13, e_184_2] at hc
    linear_combination c7174.trans k0 - hc

theorem leafCircuit_f1363 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 221)) (a (.virt 292)) (a (.virt 410)) (a (.virt 411)) := by
  have c7175 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.1
  have c7176 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.1
  have c7177 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.1
  have c7178 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7179 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7180 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7181 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7182 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7183 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7184 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7185 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7186 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7187 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7188 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7189 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7190 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7191 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_184_3 := arithEq_of_rows h (row := 184) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_3
  simp only [← c7181, ← c7182, ← c7183] at e_184_3
  have e_184_4 := arithEq_of_rows h (row := 184) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_4
  simp only [← c7184, ← c7185, ← c7186] at e_184_4
  have e_186_0 := arithEq_of_rows h (row := 186) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_0
  simp only [← c7175, k3, ← c7176, k3, ← c7177] at e_186_0
  have e_186_1 := arithEq_of_rows h (row := 186) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_1
  simp only [← c7178, ← c7179, k3, ← c7180, k3] at e_186_1
  have e_186_2 := arithEq_of_rows h (row := 186) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_2
  simp only [← c7187, ← c7188, k3, ← c7189] at e_186_2
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_184_3
    simp only [e_186_1] at hc
    linear_combination c7190.trans k0 - hc
  · have hc := e_186_2
    simp only [e_186_0, e_184_4, e_186_1] at hc
    linear_combination c7191.trans k0 - hc

theorem leafCircuit_f1364 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 221)) (a (.virt 294)) (a (.virt 412)) (a (.virt 413)) := by
  have c7192 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7193 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7194 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7195 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7196 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7197 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7198 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7199 := (leafCircuit_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7200 := (leafCircuit_copies225 a h).1
  have c7201 := (leafCircuit_copies225 a h).2.1
  have c7202 := (leafCircuit_copies225 a h).2.2.1
  have c7203 := (leafCircuit_copies225 a h).2.2.2.1
  have c7204 := (leafCircuit_copies225 a h).2.2.2.2.1
  have c7205 := (leafCircuit_copies225 a h).2.2.2.2.2.1
  have c7206 := (leafCircuit_copies225 a h).2.2.2.2.2.2.1
  have c7207 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.1
  have c7208 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_184_5 := arithEq_of_rows h (row := 184) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_5
  simp only [← c7198, ← c7199, ← c7200] at e_184_5
  have e_184_6 := arithEq_of_rows h (row := 184) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_6
  simp only [← c7201, ← c7202, ← c7203] at e_184_6
  have e_186_3 := arithEq_of_rows h (row := 186) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_3
  simp only [← c7192, k3, ← c7193, k3, ← c7194] at e_186_3
  have e_186_4 := arithEq_of_rows h (row := 186) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_4
  simp only [← c7195, ← c7196, k3, ← c7197, k5] at e_186_4
  have e_186_5 := arithEq_of_rows h (row := 186) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_5
  simp only [← c7204, ← c7205, k3, ← c7206] at e_186_5
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_184_5
    simp only [e_186_4] at hc
    linear_combination c7207.trans k0 - hc
  · have hc := e_186_5
    simp only [e_186_3, e_184_6, e_186_4] at hc
    linear_combination c7208.trans k0 - hc

theorem leafCircuit_f1365 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 221)) (a (.virt 295)) (a (.virt 414)) (a (.virt 415)) := by
  have c7209 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.1
  have c7210 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7211 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7212 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7213 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7214 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7215 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7216 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7217 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7218 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7219 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7220 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7221 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7222 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7223 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7224 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7225 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_184_7 := arithEq_of_rows h (row := 184) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_7
  simp only [← c7215, ← c7216, ← c7217] at e_184_7
  have e_184_8 := arithEq_of_rows h (row := 184) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_8
  simp only [← c7218, ← c7219, ← c7220] at e_184_8
  have e_186_6 := arithEq_of_rows h (row := 186) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_6
  simp only [← c7209, k3, ← c7210, k3, ← c7211] at e_186_6
  have e_186_7 := arithEq_of_rows h (row := 186) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_7
  simp only [← c7212, ← c7213, k3, ← c7214, k6] at e_186_7
  have e_186_8 := arithEq_of_rows h (row := 186) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_8
  simp only [← c7221, ← c7222, k3, ← c7223] at e_186_8
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_184_7
    simp only [e_186_7] at hc
    linear_combination c7224.trans k0 - hc
  · have hc := e_186_8
    simp only [e_186_6, e_184_8, e_186_7] at hc
    linear_combination c7225.trans k0 - hc

theorem leafCircuit_f1366 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 186 43) = bselect (a (.virt 408)) (a (.wire 177 55)) (a (.virt 183)) := by
  have c7226 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7227 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7228 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7229 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7230 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7231 := (leafCircuit_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_186_9 := arithEq_of_rows h (row := 186) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_9
  simp only [← c7226, ← c7227, ← c7228] at e_186_9
  have e_186_10 := arithEq_of_rows h (row := 186) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_10
  simp only [← c7229, ← c7230, ← c7231] at e_186_10
  have hr := e_186_10
  simp only [e_186_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1367 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 186 51) = bselect (a (.virt 408)) (a (.wire 182 3)) (a (.virt 184)) := by
  have c7232 := (leafCircuit_copies226 a h).1
  have c7233 := (leafCircuit_copies226 a h).2.1
  have c7234 := (leafCircuit_copies226 a h).2.2.1
  have c7235 := (leafCircuit_copies226 a h).2.2.2.1
  have c7236 := (leafCircuit_copies226 a h).2.2.2.2.1
  have c7237 := (leafCircuit_copies226 a h).2.2.2.2.2.1
  have e_186_11 := arithEq_of_rows h (row := 186) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_11
  simp only [← c7232, ← c7233, ← c7234] at e_186_11
  have e_186_12 := arithEq_of_rows h (row := 186) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_12
  simp only [← c7235, ← c7236, ← c7237] at e_186_12
  have hr := e_186_12
  simp only [e_186_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1368 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 186 59) = bselect (a (.virt 408)) (a (.wire 182 11)) (a (.virt 185)) := by
  have c7238 := (leafCircuit_copies226 a h).2.2.2.2.2.2.1
  have c7239 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.1
  have c7240 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.1
  have c7241 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.1
  have c7242 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7243 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_13 := arithEq_of_rows h (row := 186) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_13
  simp only [← c7238, ← c7239, ← c7240] at e_186_13
  have e_186_14 := arithEq_of_rows h (row := 186) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_14
  simp only [← c7241, ← c7242, ← c7243] at e_186_14
  have hr := e_186_14
  simp only [e_186_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1369 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 7) = bselect (a (.virt 408)) (a (.wire 182 19)) (a (.virt 186)) := by
  have c7244 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7245 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7246 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7247 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7248 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7249 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_187_0 := arithEq_of_rows h (row := 187) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_0
  simp only [← c7244, ← c7245, ← c7246] at e_187_0
  have e_187_1 := arithEq_of_rows h (row := 187) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_1
  simp only [← c7247, ← c7248, ← c7249] at e_187_1
  have hr := e_187_1
  simp only [e_187_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1370 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 15) = bselect (a (.virt 408)) (a (.virt 183)) (a (.virt 187)) := by
  have c7250 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7251 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7252 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7253 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7254 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7255 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_187_2 := arithEq_of_rows h (row := 187) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_2
  simp only [← c7250, ← c7251, ← c7252] at e_187_2
  have e_187_3 := arithEq_of_rows h (row := 187) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_3
  simp only [← c7253, ← c7254, ← c7255] at e_187_3
  have hr := e_187_3
  simp only [e_187_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1371 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 23) = bselect (a (.virt 410)) (a (.wire 177 55)) (a (.wire 187 15)) := by
  have c7256 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7257 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7258 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7259 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7260 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7261 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_187_4 := arithEq_of_rows h (row := 187) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_4
  simp only [← c7256, ← c7257, ← c7258] at e_187_4
  have e_187_5 := arithEq_of_rows h (row := 187) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_5
  simp only [← c7259, ← c7260, ← c7261] at e_187_5
  have hr := e_187_5
  simp only [e_187_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1372 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 31) = bselect (a (.virt 408)) (a (.virt 184)) (a (.virt 188)) := by
  have c7262 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7263 := (leafCircuit_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7264 := (leafCircuit_copies227 a h).1
  have c7265 := (leafCircuit_copies227 a h).2.1
  have c7266 := (leafCircuit_copies227 a h).2.2.1
  have c7267 := (leafCircuit_copies227 a h).2.2.2.1
  have e_187_6 := arithEq_of_rows h (row := 187) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_6
  simp only [← c7262, ← c7263, ← c7264] at e_187_6
  have e_187_7 := arithEq_of_rows h (row := 187) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_7
  simp only [← c7265, ← c7266, ← c7267] at e_187_7
  have hr := e_187_7
  simp only [e_187_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1373 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 39) = bselect (a (.virt 410)) (a (.wire 182 3)) (a (.wire 187 31)) := by
  have c7268 := (leafCircuit_copies227 a h).2.2.2.2.1
  have c7269 := (leafCircuit_copies227 a h).2.2.2.2.2.1
  have c7270 := (leafCircuit_copies227 a h).2.2.2.2.2.2.1
  have c7271 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.1
  have c7272 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.1
  have c7273 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.1
  have e_187_8 := arithEq_of_rows h (row := 187) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_8
  simp only [← c7268, ← c7269, ← c7270] at e_187_8
  have e_187_9 := arithEq_of_rows h (row := 187) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_9
  simp only [← c7271, ← c7272, ← c7273] at e_187_9
  have hr := e_187_9
  simp only [e_187_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1374 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 47) = bselect (a (.virt 408)) (a (.virt 185)) (a (.virt 189)) := by
  have c7274 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7275 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7276 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7277 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7278 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7279 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_187_10 := arithEq_of_rows h (row := 187) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_10
  simp only [← c7274, ← c7275, ← c7276] at e_187_10
  have e_187_11 := arithEq_of_rows h (row := 187) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_11
  simp only [← c7277, ← c7278, ← c7279] at e_187_11
  have hr := e_187_11
  simp only [e_187_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1375 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 187 55) = bselect (a (.virt 410)) (a (.wire 182 11)) (a (.wire 187 47)) := by
  have c7280 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7281 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7282 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7283 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7284 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7285 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_187_12 := arithEq_of_rows h (row := 187) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_12
  simp only [← c7280, ← c7281, ← c7282] at e_187_12
  have e_187_13 := arithEq_of_rows h (row := 187) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_13
  simp only [← c7283, ← c7284, ← c7285] at e_187_13
  have hr := e_187_13
  simp only [e_187_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1376 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 3) = bselect (a (.virt 408)) (a (.virt 186)) (a (.virt 190)) := by
  have c7286 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7287 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7288 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7289 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7290 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7291 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_187_14 := arithEq_of_rows h (row := 187) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_14
  simp only [← c7286, ← c7287, ← c7288] at e_187_14
  have e_188_0 := arithEq_of_rows h (row := 188) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_0
  simp only [← c7289, ← c7290, ← c7291] at e_188_0
  have hr := e_188_0
  simp only [e_187_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1377 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 11) = bselect (a (.virt 410)) (a (.wire 182 19)) (a (.wire 188 3)) := by
  have c7292 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7293 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7294 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7295 := (leafCircuit_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7296 := (leafCircuit_copies228 a h).1
  have c7297 := (leafCircuit_copies228 a h).2.1
  have e_188_1 := arithEq_of_rows h (row := 188) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_1
  simp only [← c7292, ← c7293, ← c7294] at e_188_1
  have e_188_2 := arithEq_of_rows h (row := 188) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_2
  simp only [← c7295, ← c7296, ← c7297] at e_188_2
  have hr := e_188_2
  simp only [e_188_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1378 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) := by
  have c7298 := (leafCircuit_copies228 a h).2.2.1
  have c7299 := (leafCircuit_copies228 a h).2.2.2.1
  have c7300 := (leafCircuit_copies228 a h).2.2.2.2.1
  have c7301 := (leafCircuit_copies228 a h).2.2.2.2.2.1
  have c7302 := (leafCircuit_copies228 a h).2.2.2.2.2.2.1
  have c7303 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_4 := arithEq_of_rows h (row := 164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_4
  simp only [← c7298, ← c7299, ← c7300] at e_164_4
  have e_179_10 := arithEq_of_rows h (row := 179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_10
  simp only [← c7301, ← c7302, k3, ← c7303] at e_179_10
  have hr := e_179_10
  simp only [e_164_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1379 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 19) = bselect (a (.wire 179 43)) (a (.virt 187)) (a (.virt 191)) := by
  have c7304 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.1
  have c7305 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.1
  have c7306 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7307 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7308 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7309 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_3 := arithEq_of_rows h (row := 188) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_3
  simp only [← c7304, ← c7305, ← c7306] at e_188_3
  have e_188_4 := arithEq_of_rows h (row := 188) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_4
  simp only [← c7307, ← c7308, ← c7309] at e_188_4
  have hr := e_188_4
  simp only [e_188_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1380 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 27) = bselect (a (.virt 412)) (a (.wire 177 55)) (a (.wire 188 19)) := by
  have c7310 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7311 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7312 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7313 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7314 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7315 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_5 := arithEq_of_rows h (row := 188) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_5
  simp only [← c7310, ← c7311, ← c7312] at e_188_5
  have e_188_6 := arithEq_of_rows h (row := 188) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_6
  simp only [← c7313, ← c7314, ← c7315] at e_188_6
  have hr := e_188_6
  simp only [e_188_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1381 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) := by
  have c7298 := (leafCircuit_copies228 a h).2.2.1
  have c7299 := (leafCircuit_copies228 a h).2.2.2.1
  have c7300 := (leafCircuit_copies228 a h).2.2.2.2.1
  have c7301 := (leafCircuit_copies228 a h).2.2.2.2.2.1
  have c7302 := (leafCircuit_copies228 a h).2.2.2.2.2.2.1
  have c7303 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_4 := arithEq_of_rows h (row := 164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_4
  simp only [← c7298, ← c7299, ← c7300] at e_164_4
  have e_179_10 := arithEq_of_rows h (row := 179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_10
  simp only [← c7301, ← c7302, k3, ← c7303] at e_179_10
  have hr := e_179_10
  simp only [e_164_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1382 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 35) = bselect (a (.wire 179 43)) (a (.virt 188)) (a (.virt 192)) := by
  have c7316 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7317 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7318 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7319 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7320 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7321 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_7 := arithEq_of_rows h (row := 188) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_7
  simp only [← c7316, ← c7317, ← c7318] at e_188_7
  have e_188_8 := arithEq_of_rows h (row := 188) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_8
  simp only [← c7319, ← c7320, ← c7321] at e_188_8
  have hr := e_188_8
  simp only [e_188_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1383 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 43) = bselect (a (.virt 412)) (a (.wire 182 3)) (a (.wire 188 35)) := by
  have c7322 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7323 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7324 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7325 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7326 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7327 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_188_9 := arithEq_of_rows h (row := 188) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_9
  simp only [← c7322, ← c7323, ← c7324] at e_188_9
  have e_188_10 := arithEq_of_rows h (row := 188) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_10
  simp only [← c7325, ← c7326, ← c7327] at e_188_10
  have hr := e_188_10
  simp only [e_188_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1384 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) := by
  have c7298 := (leafCircuit_copies228 a h).2.2.1
  have c7299 := (leafCircuit_copies228 a h).2.2.2.1
  have c7300 := (leafCircuit_copies228 a h).2.2.2.2.1
  have c7301 := (leafCircuit_copies228 a h).2.2.2.2.2.1
  have c7302 := (leafCircuit_copies228 a h).2.2.2.2.2.2.1
  have c7303 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_4 := arithEq_of_rows h (row := 164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_4
  simp only [← c7298, ← c7299, ← c7300] at e_164_4
  have e_179_10 := arithEq_of_rows h (row := 179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_10
  simp only [← c7301, ← c7302, k3, ← c7303] at e_179_10
  have hr := e_179_10
  simp only [e_164_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1385 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 51) = bselect (a (.wire 179 43)) (a (.virt 189)) (a (.virt 193)) := by
  have c7328 := (leafCircuit_copies229 a h).1
  have c7329 := (leafCircuit_copies229 a h).2.1
  have c7330 := (leafCircuit_copies229 a h).2.2.1
  have c7331 := (leafCircuit_copies229 a h).2.2.2.1
  have c7332 := (leafCircuit_copies229 a h).2.2.2.2.1
  have c7333 := (leafCircuit_copies229 a h).2.2.2.2.2.1
  have e_188_11 := arithEq_of_rows h (row := 188) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_11
  simp only [← c7328, ← c7329, ← c7330] at e_188_11
  have e_188_12 := arithEq_of_rows h (row := 188) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_12
  simp only [← c7331, ← c7332, ← c7333] at e_188_12
  have hr := e_188_12
  simp only [e_188_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1386 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 188 59) = bselect (a (.virt 412)) (a (.wire 182 11)) (a (.wire 188 51)) := by
  have c7334 := (leafCircuit_copies229 a h).2.2.2.2.2.2.1
  have c7335 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.1
  have c7336 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.1
  have c7337 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.1
  have c7338 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7339 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_13 := arithEq_of_rows h (row := 188) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_13
  simp only [← c7334, ← c7335, ← c7336] at e_188_13
  have e_188_14 := arithEq_of_rows h (row := 188) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_14
  simp only [← c7337, ← c7338, ← c7339] at e_188_14
  have hr := e_188_14
  simp only [e_188_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1387 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) := by
  have c7298 := (leafCircuit_copies228 a h).2.2.1
  have c7299 := (leafCircuit_copies228 a h).2.2.2.1
  have c7300 := (leafCircuit_copies228 a h).2.2.2.2.1
  have c7301 := (leafCircuit_copies228 a h).2.2.2.2.2.1
  have c7302 := (leafCircuit_copies228 a h).2.2.2.2.2.2.1
  have c7303 := (leafCircuit_copies228 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_4 := arithEq_of_rows h (row := 164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_4
  simp only [← c7298, ← c7299, ← c7300] at e_164_4
  have e_179_10 := arithEq_of_rows h (row := 179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_10
  simp only [← c7301, ← c7302, k3, ← c7303] at e_179_10
  have hr := e_179_10
  simp only [e_164_4] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1388 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 7) = bselect (a (.wire 179 43)) (a (.virt 190)) (a (.virt 194)) := by
  have c7340 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7341 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7342 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7343 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7344 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7345 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_189_0 := arithEq_of_rows h (row := 189) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_0
  simp only [← c7340, ← c7341, ← c7342] at e_189_0
  have e_189_1 := arithEq_of_rows h (row := 189) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_1
  simp only [← c7343, ← c7344, ← c7345] at e_189_1
  have hr := e_189_1
  simp only [e_189_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1389 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 15) = bselect (a (.virt 412)) (a (.wire 182 19)) (a (.wire 189 7)) := by
  have c7346 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7347 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7348 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7349 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7350 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7351 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_189_2 := arithEq_of_rows h (row := 189) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_2
  simp only [← c7346, ← c7347, ← c7348] at e_189_2
  have e_189_3 := arithEq_of_rows h (row := 189) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_3
  simp only [← c7349, ← c7350, ← c7351] at e_189_3
  have hr := e_189_3
  simp only [e_189_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1390 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 23) = bselect (a (.virt 414)) (a (.wire 177 55)) (a (.virt 191)) := by
  have c7352 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7353 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7354 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7355 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7356 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7357 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_189_4 := arithEq_of_rows h (row := 189) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_4
  simp only [← c7352, ← c7353, ← c7354] at e_189_4
  have e_189_5 := arithEq_of_rows h (row := 189) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_5
  simp only [← c7355, ← c7356, ← c7357] at e_189_5
  have hr := e_189_5
  simp only [e_189_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1391 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 31) = bselect (a (.virt 414)) (a (.wire 182 3)) (a (.virt 192)) := by
  have c7358 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7359 := (leafCircuit_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7360 := (leafCircuit_copies230 a h).1
  have c7361 := (leafCircuit_copies230 a h).2.1
  have c7362 := (leafCircuit_copies230 a h).2.2.1
  have c7363 := (leafCircuit_copies230 a h).2.2.2.1
  have e_189_6 := arithEq_of_rows h (row := 189) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_6
  simp only [← c7358, ← c7359, ← c7360] at e_189_6
  have e_189_7 := arithEq_of_rows h (row := 189) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_7
  simp only [← c7361, ← c7362, ← c7363] at e_189_7
  have hr := e_189_7
  simp only [e_189_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1392 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 39) = bselect (a (.virt 414)) (a (.wire 182 11)) (a (.virt 193)) := by
  have c7364 := (leafCircuit_copies230 a h).2.2.2.2.1
  have c7365 := (leafCircuit_copies230 a h).2.2.2.2.2.1
  have c7366 := (leafCircuit_copies230 a h).2.2.2.2.2.2.1
  have c7367 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.1
  have c7368 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.1
  have c7369 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.1
  have e_189_8 := arithEq_of_rows h (row := 189) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_8
  simp only [← c7364, ← c7365, ← c7366] at e_189_8
  have e_189_9 := arithEq_of_rows h (row := 189) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_9
  simp only [← c7367, ← c7368, ← c7369] at e_189_9
  have hr := e_189_9
  simp only [e_189_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1393 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 47) = bselect (a (.virt 414)) (a (.wire 182 19)) (a (.virt 194)) := by
  have c7370 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7371 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7372 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7373 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7374 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7375 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_189_10 := arithEq_of_rows h (row := 189) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_10
  simp only [← c7370, ← c7371, ← c7372] at e_189_10
  have e_189_11 := arithEq_of_rows h (row := 189) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_11
  simp only [← c7373, ← c7374, ← c7375] at e_189_11
  have hr := e_189_11
  simp only [e_189_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1394 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 193 12) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 0 ∧ a (.wire 193 13) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 1 ∧ a (.wire 193 14) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 2 ∧ a (.wire 193 15) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 3) := by
  have c7376 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7377 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7378 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7379 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7380 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7381 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7382 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7383 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7384 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7385 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7386 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7387 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7388 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7389 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7390 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7391 := (leafCircuit_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7392 := (leafCircuit_copies231 a h).1
  have c7393 := (leafCircuit_copies231 a h).2.1
  have c7394 := (leafCircuit_copies231 a h).2.2.1
  have c7395 := (leafCircuit_copies231 a h).2.2.2.1
  have c7396 := (leafCircuit_copies231 a h).2.2.2.2.1
  have c7397 := (leafCircuit_copies231 a h).2.2.2.2.2.1
  have c7398 := (leafCircuit_copies231 a h).2.2.2.2.2.2.1
  have c7399 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.1
  have c7400 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.1
  have c7401 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.1
  have c7402 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7403 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7404 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7405 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7406 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7407 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7408 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7409 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7410 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7411 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7412 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7413 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7414 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7415 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7416 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7417 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7418 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7419 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7420 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7421 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7422 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7423 := (leafCircuit_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7424 := (leafCircuit_copies232 a h).1
  have c7425 := (leafCircuit_copies232 a h).2.1
  have c7426 := (leafCircuit_copies232 a h).2.2.1
  have c7427 := (leafCircuit_copies232 a h).2.2.2.1
  have c7428 := (leafCircuit_copies232 a h).2.2.2.2.1
  have c7429 := (leafCircuit_copies232 a h).2.2.2.2.2.1
  have c7430 := (leafCircuit_copies232 a h).2.2.2.2.2.2.1
  have c7431 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.1
  have c7432 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.1
  have c7433 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.1
  have c7434 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7435 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7436 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7437 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7438 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_179_11 := arithEq_of_rows h (row := 179) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_11
  simp only [← c7388, ← c7389, k3, ← c7390] at e_179_11
  have e_179_12 := arithEq_of_rows h (row := 179) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_12
  simp only [← c7391, ← c7392, k3, ← c7393] at e_179_12
  have e_179_13 := arithEq_of_rows h (row := 179) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_13
  simp only [← c7394, ← c7395, k3, ← c7396] at e_179_13
  have e_179_14 := arithEq_of_rows h (row := 179) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_14
  simp only [← c7397, ← c7398, k3, ← c7399] at e_179_14
  have e_191_0 := arithEq_of_rows h (row := 191) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_0
  simp only [← c7400, ← c7401, k3, ← c7402] at e_191_0
  have e_191_1 := arithEq_of_rows h (row := 191) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_1
  simp only [← c7403, ← c7404, k3, ← c7405] at e_191_1
  have e_191_2 := arithEq_of_rows h (row := 191) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_2
  simp only [← c7406, ← c7407, k3, ← c7408] at e_191_2
  have e_191_3 := arithEq_of_rows h (row := 191) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_3
  simp only [← c7409, ← c7410, k3, ← c7411] at e_191_3
  have e_191_4 := arithEq_of_rows h (row := 191) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_4
  simp only [← c7424, ← c7425, k3, ← c7426, k3] at e_191_4
  have hin0 : poseidon2In a 190 = addBlock (fun _ => 0) [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11)] :=
    poseidon2In_first c7376.symm c7377.symm c7378.symm c7379.symm c7380.symm c7381.symm c7382.symm c7383.symm (c7384.symm.trans k0) (c7385.symm.trans k0) (c7386.symm.trans k0) (c7387.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 190) rfl rfl hin0
  have hin1 : poseidon2In a 192 = addBlock (poseidon2Out a 190) [a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] :=
    poseidon2In_chain (by linear_combination -c7412 + e_179_11) (by linear_combination -c7413 + e_179_12) (by linear_combination -c7414 + e_179_13) (by linear_combination -c7415 + e_179_14) (by linear_combination -c7416 + e_191_0) (by linear_combination -c7417 + e_191_1) (by linear_combination -c7418 + e_191_2) (by linear_combination -c7419 + e_191_3) c7420.symm c7421.symm c7422.symm c7423.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 192) rfl rfl hin1
  have hin2 : poseidon2In a 193 = addBlock (poseidon2Out a 192) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c7427 + e_191_4) (c7428.symm.trans (add_zero _).symm) (c7429.symm.trans (add_zero _).symm) (c7430.symm.trans (add_zero _).symm) (c7431.symm.trans (add_zero _).symm) (c7432.symm.trans (add_zero _).symm) (c7433.symm.trans (add_zero _).symm) (c7434.symm.trans (add_zero _).symm) c7435.symm c7436.symm c7437.symm c7438.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 193) rfl rfl hin2
  have hpad : pad10 [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] =
      [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11)] ++ ([a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1395 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 189 55) = bselect (a (.wire 179 39)) (a (.wire 193 12)) (a (.wire 177 55)) := by
  have c7439 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7440 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7441 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7442 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7443 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7444 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_189_12 := arithEq_of_rows h (row := 189) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_12
  simp only [← c7439, ← c7440, ← c7441] at e_189_12
  have e_189_13 := arithEq_of_rows h (row := 189) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_13
  simp only [← c7442, ← c7443, ← c7444] at e_189_13
  have hr := e_189_13
  simp only [e_189_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1396 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 3) = bselect (a (.wire 179 39)) (a (.wire 193 13)) (a (.wire 182 3)) := by
  have c7445 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7446 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7447 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7448 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7449 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7450 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_189_14 := arithEq_of_rows h (row := 189) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_14
  simp only [← c7445, ← c7446, ← c7447] at e_189_14
  have e_194_0 := arithEq_of_rows h (row := 194) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_0
  simp only [← c7448, ← c7449, ← c7450] at e_194_0
  have hr := e_194_0
  simp only [e_189_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1397 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 11) = bselect (a (.wire 179 39)) (a (.wire 193 14)) (a (.wire 182 11)) := by
  have c7451 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7452 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7453 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7454 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7455 := (leafCircuit_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7456 := (leafCircuit_copies233 a h).1
  have e_194_1 := arithEq_of_rows h (row := 194) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_1
  simp only [← c7451, ← c7452, ← c7453] at e_194_1
  have e_194_2 := arithEq_of_rows h (row := 194) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_2
  simp only [← c7454, ← c7455, ← c7456] at e_194_2
  have hr := e_194_2
  simp only [e_194_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1398 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 19) = bselect (a (.wire 179 39)) (a (.wire 193 15)) (a (.wire 182 19)) := by
  have c7457 := (leafCircuit_copies233 a h).2.1
  have c7458 := (leafCircuit_copies233 a h).2.2.1
  have c7459 := (leafCircuit_copies233 a h).2.2.2.1
  have c7460 := (leafCircuit_copies233 a h).2.2.2.2.1
  have c7461 := (leafCircuit_copies233 a h).2.2.2.2.2.1
  have c7462 := (leafCircuit_copies233 a h).2.2.2.2.2.2.1
  have e_194_3 := arithEq_of_rows h (row := 194) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_3
  simp only [← c7457, ← c7458, ← c7459] at e_194_3
  have e_194_4 := arithEq_of_rows h (row := 194) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_4
  simp only [← c7460, ← c7461, ← c7462] at e_194_4
  have hr := e_194_4
  simp only [e_194_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1399 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    BaseSum 2 (a (.virt 12)) [a (.wire 195 1), a (.wire 195 2), a (.wire 195 3), a (.wire 195 4), a (.wire 195 5)] := by
  have c7463 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.1
  have c7464 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.1
  have c7465 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.1
  have c7466 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7467 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7468 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7469 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7470 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7471 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7472 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7473 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7474 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7475 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7476 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7477 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7478 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7479 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7480 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7481 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7482 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7483 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7484 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7485 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7486 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7487 := (leafCircuit_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7488 := (leafCircuit_copies234 a h).1
  have c7489 := (leafCircuit_copies234 a h).2.1
  have c7490 := (leafCircuit_copies234 a h).2.2.1
  have c7491 := (leafCircuit_copies234 a h).2.2.2.1
  have c7492 := (leafCircuit_copies234 a h).2.2.2.2.1
  have c7493 := (leafCircuit_copies234 a h).2.2.2.2.2.1
  have c7494 := (leafCircuit_copies234 a h).2.2.2.2.2.2.1
  have c7495 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.1
  have c7496 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.1
  have c7497 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.1
  have c7498 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7499 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7500 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7501 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7502 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7503 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7504 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7505 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7506 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7507 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7508 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7509 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7510 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7511 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7512 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7513 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7514 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7515 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7516 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7517 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := baseSum_of_row h (row := 195) (N := 59) (n := 5) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c7463.trans k0
    · exact c7464.trans k0
    · exact c7465.trans k0
    · exact c7466.trans k0
    · exact c7467.trans k0
    · exact c7468.trans k0
    · exact c7469.trans k0
    · exact c7470.trans k0
    · exact c7471.trans k0
    · exact c7472.trans k0
    · exact c7473.trans k0
    · exact c7474.trans k0
    · exact c7475.trans k0
    · exact c7476.trans k0
    · exact c7477.trans k0
    · exact c7478.trans k0
    · exact c7479.trans k0
    · exact c7480.trans k0
    · exact c7481.trans k0
    · exact c7482.trans k0
    · exact c7483.trans k0
    · exact c7484.trans k0
    · exact c7485.trans k0
    · exact c7486.trans k0
    · exact c7487.trans k0
    · exact c7488.trans k0
    · exact c7489.trans k0
    · exact c7490.trans k0
    · exact c7491.trans k0
    · exact c7492.trans k0
    · exact c7493.trans k0
    · exact c7494.trans k0
    · exact c7495.trans k0
    · exact c7496.trans k0
    · exact c7497.trans k0
    · exact c7498.trans k0
    · exact c7499.trans k0
    · exact c7500.trans k0
    · exact c7501.trans k0
    · exact c7502.trans k0
    · exact c7503.trans k0
    · exact c7504.trans k0
    · exact c7505.trans k0
    · exact c7506.trans k0
    · exact c7507.trans k0
    · exact c7508.trans k0
    · exact c7509.trans k0
    · exact c7510.trans k0
    · exact c7511.trans k0
    · exact c7512.trans k0
    · exact c7513.trans k0
    · exact c7514.trans k0
    · exact c7515.trans k0
    · exact c7516.trans k0
    )
  rwa [c7517] at hr

theorem leafCircuit_f1400 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 292) = bnot (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1401 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = band (a (.virt 292)) (a (.wire 195 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1402 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = band (a (.wire 195 5)) (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1403 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = bor (a (.virt 14)) (a (.wire 195 5)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1404 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 14) * a (.wire 195 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1405 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = a (.virt 294) * a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0, k5]
  ring

theorem leafCircuit_f1406 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = a (.virt 14) + a (.wire 195 5) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1407 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = a (.wire 195 5) - a (.virt 14) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k0]
  ring

theorem leafCircuit_f1408 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 23) = bnot (a (.wire 195 5)) := by
  have c7518 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7519 := (leafCircuit_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7520 := (leafCircuit_copies235 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_5 := arithEq_of_rows h (row := 194) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_5
  simp only [← c7518, k3, ← c7519, k3, ← c7520] at e_194_5
  have hr := e_194_5
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1409 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 23) = band (a (.virt 292)) (a (.wire 194 23)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k3]
  ring

theorem leafCircuit_f1410 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1411 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 4)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1412 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 194 23)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1413 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1414 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 4) = a (.virt 292) * a (.wire 195 4) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1415 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 39) = a (.virt 294) * a (.wire 195 4) := by
  have c7521 := (leafCircuit_copies235 a h).2.1
  have c7522 := (leafCircuit_copies235 a h).2.2.1
  have c7523 := (leafCircuit_copies235 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_184_9 := arithEq_of_rows h (row := 184) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_9
  simp only [← c7521, k5, ← c7522, ← c7523, k5] at e_184_9
  have hr := e_184_9
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1416 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 23) = a (.virt 292) + a (.wire 195 4) := by
  have c7524 := (leafCircuit_copies235 a h).2.2.2.2.1
  have c7525 := (leafCircuit_copies235 a h).2.2.2.2.2.1
  have c7526 := (leafCircuit_copies235 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_191_5 := arithEq_of_rows h (row := 191) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_5
  simp only [← c7524, k3, ← c7525, k3, ← c7526] at e_191_5
  have hr := e_191_5
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1417 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 27) = a (.wire 191 23) - a (.wire 184 39) := by
  have c7527 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.1
  have c7528 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.1
  have c7529 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_6 := arithEq_of_rows h (row := 194) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_6
  simp only [← c7527, ← c7528, k3, ← c7529] at e_194_6
  have hr := e_194_6
  linear_combination hr

theorem leafCircuit_f1418 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 31) = bnot (a (.wire 194 27)) := by
  have c7530 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7531 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7532 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_7 := arithEq_of_rows h (row := 194) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_7
  simp only [← c7530, k3, ← c7531, k3, ← c7532] at e_194_7
  have hr := e_194_7
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1419 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 43) = band (a (.wire 194 23)) (a (.wire 194 31)) := by
  have c7533 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7534 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7535 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_184_10 := arithEq_of_rows h (row := 184) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_10
  simp only [← c7533, ← c7534, ← c7535] at e_184_10
  have hr := e_184_10
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1420 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1421 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 3)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1422 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 184 43)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1423 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1424 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 3) = a (.virt 292) * a (.wire 195 3) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1425 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 47) = a (.virt 294) * a (.wire 195 3) := by
  have c7536 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7537 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7538 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_184_11 := arithEq_of_rows h (row := 184) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_11
  simp only [← c7536, k5, ← c7537, ← c7538, k5] at e_184_11
  have hr := e_184_11
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1426 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 27) = a (.virt 292) + a (.wire 195 3) := by
  have c7539 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7540 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7541 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_191_6 := arithEq_of_rows h (row := 191) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_6
  simp only [← c7539, k3, ← c7540, k3, ← c7541] at e_191_6
  have hr := e_191_6
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1427 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 35) = a (.wire 191 27) - a (.wire 184 47) := by
  have c7542 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7543 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7544 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_8 := arithEq_of_rows h (row := 194) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_8
  simp only [← c7542, ← c7543, k3, ← c7544] at e_194_8
  have hr := e_194_8
  linear_combination hr

theorem leafCircuit_f1428 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 39) = bnot (a (.wire 194 35)) := by
  have c7545 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7546 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7547 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_9 := arithEq_of_rows h (row := 194) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_9
  simp only [← c7545, k3, ← c7546, k3, ← c7547] at e_194_9
  have hr := e_194_9
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1429 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 51) = band (a (.wire 184 43)) (a (.wire 194 39)) := by
  have c7548 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7549 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7550 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_184_12 := arithEq_of_rows h (row := 184) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_12
  simp only [← c7548, ← c7549, ← c7550] at e_184_12
  have hr := e_184_12
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1430 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1431 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 2)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1432 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 184 51)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1433 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1434 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 2) = a (.virt 292) * a (.wire 195 2) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1435 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 55) = a (.virt 294) * a (.wire 195 2) := by
  have c7551 := (leafCircuit_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7552 := (leafCircuit_copies236 a h).1
  have c7553 := (leafCircuit_copies236 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_184_13 := arithEq_of_rows h (row := 184) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_13
  simp only [← c7551, k5, ← c7552, ← c7553, k5] at e_184_13
  have hr := e_184_13
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1436 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 31) = a (.virt 292) + a (.wire 195 2) := by
  have c7554 := (leafCircuit_copies236 a h).2.2.1
  have c7555 := (leafCircuit_copies236 a h).2.2.2.1
  have c7556 := (leafCircuit_copies236 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_191_7 := arithEq_of_rows h (row := 191) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_7
  simp only [← c7554, k3, ← c7555, k3, ← c7556] at e_191_7
  have hr := e_191_7
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1437 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 43) = a (.wire 191 31) - a (.wire 184 55) := by
  have c7557 := (leafCircuit_copies236 a h).2.2.2.2.2.1
  have c7558 := (leafCircuit_copies236 a h).2.2.2.2.2.2.1
  have c7559 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_10 := arithEq_of_rows h (row := 194) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_10
  simp only [← c7557, ← c7558, k3, ← c7559] at e_194_10
  have hr := e_194_10
  linear_combination hr

theorem leafCircuit_f1438 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 47) = bnot (a (.wire 194 43)) := by
  have c7560 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.1
  have c7561 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.1
  have c7562 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_11 := arithEq_of_rows h (row := 194) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_11
  simp only [← c7560, k3, ← c7561, k3, ← c7562] at e_194_11
  have hr := e_194_11
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1439 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 184 59) = band (a (.wire 184 51)) (a (.wire 194 47)) := by
  have c7563 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7564 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7565 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_184_14 := arithEq_of_rows h (row := 184) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_14
  simp only [← c7563, ← c7564, ← c7565] at e_184_14
  have hr := e_184_14
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1440 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = bnot (a (.virt 292)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bnot, k0, k3]
  ring

theorem leafCircuit_f1441 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 1)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1442 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 14) = band (a (.virt 14)) (a (.wire 184 59)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [band, k0]
  ring

theorem leafCircuit_f1443 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [bor, k0]
  ring

theorem leafCircuit_f1444 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 195 1) = a (.virt 292) * a (.wire 195 1) := by
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  simp only [k3]
  ring

theorem leafCircuit_f1445 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 3) = a (.virt 294) * a (.wire 195 1) := by
  have c7566 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7567 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7568 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_196_0 := arithEq_of_rows h (row := 196) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_0
  simp only [← c7566, k5, ← c7567, ← c7568, k5] at e_196_0
  have hr := e_196_0
  simp only [k5]
  linear_combination hr

theorem leafCircuit_f1446 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 35) = a (.virt 292) + a (.wire 195 1) := by
  have c7569 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7570 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7571 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_191_8 := arithEq_of_rows h (row := 191) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_8
  simp only [← c7569, k3, ← c7570, k3, ← c7571] at e_191_8
  have hr := e_191_8
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1447 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 51) = a (.wire 191 35) - a (.wire 196 3) := by
  have c7572 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7573 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7574 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_12 := arithEq_of_rows h (row := 194) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_12
  simp only [← c7572, ← c7573, k3, ← c7574] at e_194_12
  have hr := e_194_12
  linear_combination hr

theorem leafCircuit_f1448 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 194 55) = bnot (a (.wire 194 51)) := by
  have c7575 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7576 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7577 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_13 := arithEq_of_rows h (row := 194) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_13
  simp only [← c7575, k3, ← c7576, k3, ← c7577] at e_194_13
  have hr := e_194_13
  simp only [bnot]
  linear_combination hr

theorem leafCircuit_f1449 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 7) = band (a (.wire 184 59)) (a (.wire 194 55)) := by
  have c7578 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7579 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7580 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_196_1 := arithEq_of_rows h (row := 196) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_1
  simp only [← c7578, ← c7579, ← c7580] at e_196_1
  have hr := e_196_1
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1450 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 222)) 2 := by
  have c7581 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7582 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7583 := (leafCircuit_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7584 := (leafCircuit_copies237 a h).1
  have c7585 := (leafCircuit_copies237 a h).2.1
  have c7586 := (leafCircuit_copies237 a h).2.2.1
  have c7587 := (leafCircuit_copies237 a h).2.2.2.1
  have c7588 := (leafCircuit_copies237 a h).2.2.2.2.1
  have c7589 := (leafCircuit_copies237 a h).2.2.2.2.2.1
  have c7590 := (leafCircuit_copies237 a h).2.2.2.2.2.2.1
  have c7591 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.1
  have c7592 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.1
  have c7593 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.1
  have c7594 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7595 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7596 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7597 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7598 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7599 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7600 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7601 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7602 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7603 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7604 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7605 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7606 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7607 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7608 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7609 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7610 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7611 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7612 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7613 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7614 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7615 := (leafCircuit_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7616 := (leafCircuit_copies238 a h).1
  have c7617 := (leafCircuit_copies238 a h).2.1
  have c7618 := (leafCircuit_copies238 a h).2.2.1
  have c7619 := (leafCircuit_copies238 a h).2.2.2.1
  have c7620 := (leafCircuit_copies238 a h).2.2.2.2.1
  have c7621 := (leafCircuit_copies238 a h).2.2.2.2.2.1
  have c7622 := (leafCircuit_copies238 a h).2.2.2.2.2.2.1
  have c7623 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.1
  have c7624 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.1
  have c7625 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.1
  have c7626 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7627 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7628 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7629 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7630 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7631 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7632 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7633 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7634 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7635 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7636 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7637 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7638 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 197) (N := 59) (n := 2) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c7581.trans k0
    · exact c7582.trans k0
    · exact c7583.trans k0
    · exact c7584.trans k0
    · exact c7585.trans k0
    · exact c7586.trans k0
    · exact c7587.trans k0
    · exact c7588.trans k0
    · exact c7589.trans k0
    · exact c7590.trans k0
    · exact c7591.trans k0
    · exact c7592.trans k0
    · exact c7593.trans k0
    · exact c7594.trans k0
    · exact c7595.trans k0
    · exact c7596.trans k0
    · exact c7597.trans k0
    · exact c7598.trans k0
    · exact c7599.trans k0
    · exact c7600.trans k0
    · exact c7601.trans k0
    · exact c7602.trans k0
    · exact c7603.trans k0
    · exact c7604.trans k0
    · exact c7605.trans k0
    · exact c7606.trans k0
    · exact c7607.trans k0
    · exact c7608.trans k0
    · exact c7609.trans k0
    · exact c7610.trans k0
    · exact c7611.trans k0
    · exact c7612.trans k0
    · exact c7613.trans k0
    · exact c7614.trans k0
    · exact c7615.trans k0
    · exact c7616.trans k0
    · exact c7617.trans k0
    · exact c7618.trans k0
    · exact c7619.trans k0
    · exact c7620.trans k0
    · exact c7621.trans k0
    · exact c7622.trans k0
    · exact c7623.trans k0
    · exact c7624.trans k0
    · exact c7625.trans k0
    · exact c7626.trans k0
    · exact c7627.trans k0
    · exact c7628.trans k0
    · exact c7629.trans k0
    · exact c7630.trans k0
    · exact c7631.trans k0
    · exact c7632.trans k0
    · exact c7633.trans k0
    · exact c7634.trans k0
    · exact c7635.trans k0
    · exact c7636.trans k0
    · exact c7637.trans k0
    )
  rwa [c7638] at hr

theorem leafCircuit_f1451 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 222)) (a (.virt 14)) (a (.virt 416)) (a (.virt 417)) := by
  have c7639 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7640 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7641 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7642 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7643 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7644 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7645 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7646 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7647 := (leafCircuit_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7648 := (leafCircuit_copies239 a h).1
  have c7649 := (leafCircuit_copies239 a h).2.1
  have c7650 := (leafCircuit_copies239 a h).2.2.1
  have c7651 := (leafCircuit_copies239 a h).2.2.2.1
  have c7652 := (leafCircuit_copies239 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_194_14 := arithEq_of_rows h (row := 194) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_14
  simp only [← c7639, k3, ← c7640, k3, ← c7641] at e_194_14
  have e_196_2 := arithEq_of_rows h (row := 196) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_2
  simp only [← c7642, ← c7643, ← c7644] at e_196_2
  have e_196_3 := arithEq_of_rows h (row := 196) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_3
  simp only [← c7645, ← c7646, ← c7647] at e_196_3
  have e_198_0 := arithEq_of_rows h (row := 198) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_0
  simp only [← c7648, ← c7649, k3, ← c7650] at e_198_0
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_196_2
    linear_combination c7651.trans k0 - hc
  · have hc := e_198_0
    simp only [e_194_14, e_196_3] at hc
    linear_combination c7652.trans k0 - hc

theorem leafCircuit_f1452 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 222)) (a (.virt 292)) (a (.virt 418)) (a (.virt 419)) := by
  have c7653 := (leafCircuit_copies239 a h).2.2.2.2.2.1
  have c7654 := (leafCircuit_copies239 a h).2.2.2.2.2.2.1
  have c7655 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.1
  have c7656 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.1
  have c7657 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.1
  have c7658 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7659 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7660 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7661 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7662 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7663 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7664 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7665 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7666 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7667 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7668 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7669 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_196_4 := arithEq_of_rows h (row := 196) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_4
  simp only [← c7659, ← c7660, ← c7661] at e_196_4
  have e_196_5 := arithEq_of_rows h (row := 196) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_5
  simp only [← c7662, ← c7663, ← c7664] at e_196_5
  have e_198_1 := arithEq_of_rows h (row := 198) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_1
  simp only [← c7653, k3, ← c7654, k3, ← c7655] at e_198_1
  have e_198_2 := arithEq_of_rows h (row := 198) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_2
  simp only [← c7656, ← c7657, k3, ← c7658, k3] at e_198_2
  have e_198_3 := arithEq_of_rows h (row := 198) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_3
  simp only [← c7665, ← c7666, k3, ← c7667] at e_198_3
  simp only [k3]
  refine ⟨?_, ?_⟩
  · have hc := e_196_4
    simp only [e_198_2] at hc
    linear_combination c7668.trans k0 - hc
  · have hc := e_198_3
    simp only [e_198_1, e_196_5, e_198_2] at hc
    linear_combination c7669.trans k0 - hc

theorem leafCircuit_f1453 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 222)) (a (.virt 294)) (a (.virt 420)) (a (.virt 421)) := by
  have c7670 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7671 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7672 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7673 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7674 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7675 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7676 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7677 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7678 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7679 := (leafCircuit_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7680 := (leafCircuit_copies240 a h).1
  have c7681 := (leafCircuit_copies240 a h).2.1
  have c7682 := (leafCircuit_copies240 a h).2.2.1
  have c7683 := (leafCircuit_copies240 a h).2.2.2.1
  have c7684 := (leafCircuit_copies240 a h).2.2.2.2.1
  have c7685 := (leafCircuit_copies240 a h).2.2.2.2.2.1
  have c7686 := (leafCircuit_copies240 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_196_6 := arithEq_of_rows h (row := 196) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_6
  simp only [← c7676, ← c7677, ← c7678] at e_196_6
  have e_196_7 := arithEq_of_rows h (row := 196) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_7
  simp only [← c7679, ← c7680, ← c7681] at e_196_7
  have e_198_4 := arithEq_of_rows h (row := 198) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_4
  simp only [← c7670, k3, ← c7671, k3, ← c7672] at e_198_4
  have e_198_5 := arithEq_of_rows h (row := 198) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_5
  simp only [← c7673, ← c7674, k3, ← c7675, k5] at e_198_5
  have e_198_6 := arithEq_of_rows h (row := 198) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_6
  simp only [← c7682, ← c7683, k3, ← c7684] at e_198_6
  simp only [k5]
  refine ⟨?_, ?_⟩
  · have hc := e_196_6
    simp only [e_198_5] at hc
    linear_combination c7685.trans k0 - hc
  · have hc := e_198_6
    simp only [e_198_4, e_196_7, e_198_5] at hc
    linear_combination c7686.trans k0 - hc

theorem leafCircuit_f1454 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 222)) (a (.virt 295)) (a (.virt 422)) (a (.virt 423)) := by
  have c7687 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.1
  have c7688 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.1
  have c7689 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.1
  have c7690 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7691 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7692 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7693 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7694 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7695 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7696 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7697 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7698 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7699 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7700 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7701 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7702 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7703 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_196_8 := arithEq_of_rows h (row := 196) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_8
  simp only [← c7693, ← c7694, ← c7695] at e_196_8
  have e_196_9 := arithEq_of_rows h (row := 196) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_9
  simp only [← c7696, ← c7697, ← c7698] at e_196_9
  have e_198_7 := arithEq_of_rows h (row := 198) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_7
  simp only [← c7687, k3, ← c7688, k3, ← c7689] at e_198_7
  have e_198_8 := arithEq_of_rows h (row := 198) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_8
  simp only [← c7690, ← c7691, k3, ← c7692, k6] at e_198_8
  have e_198_9 := arithEq_of_rows h (row := 198) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_9
  simp only [← c7699, ← c7700, k3, ← c7701] at e_198_9
  simp only [k6]
  refine ⟨?_, ?_⟩
  · have hc := e_196_8
    simp only [e_198_8] at hc
    linear_combination c7702.trans k0 - hc
  · have hc := e_198_9
    simp only [e_198_7, e_196_9, e_198_8] at hc
    linear_combination c7703.trans k0 - hc

theorem leafCircuit_f1455 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 198 47) = bselect (a (.virt 416)) (a (.wire 189 55)) (a (.virt 195)) := by
  have c7704 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7705 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7706 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7707 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7708 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7709 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_198_10 := arithEq_of_rows h (row := 198) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_10
  simp only [← c7704, ← c7705, ← c7706] at e_198_10
  have e_198_11 := arithEq_of_rows h (row := 198) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_11
  simp only [← c7707, ← c7708, ← c7709] at e_198_11
  have hr := e_198_11
  simp only [e_198_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1456 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 198 55) = bselect (a (.virt 416)) (a (.wire 194 3)) (a (.virt 196)) := by
  have c7710 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7711 := (leafCircuit_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7712 := (leafCircuit_copies241 a h).1
  have c7713 := (leafCircuit_copies241 a h).2.1
  have c7714 := (leafCircuit_copies241 a h).2.2.1
  have c7715 := (leafCircuit_copies241 a h).2.2.2.1
  have e_198_12 := arithEq_of_rows h (row := 198) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_12
  simp only [← c7710, ← c7711, ← c7712] at e_198_12
  have e_198_13 := arithEq_of_rows h (row := 198) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_13
  simp only [← c7713, ← c7714, ← c7715] at e_198_13
  have hr := e_198_13
  simp only [e_198_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1457 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 3) = bselect (a (.virt 416)) (a (.wire 194 11)) (a (.virt 197)) := by
  have c7716 := (leafCircuit_copies241 a h).2.2.2.2.1
  have c7717 := (leafCircuit_copies241 a h).2.2.2.2.2.1
  have c7718 := (leafCircuit_copies241 a h).2.2.2.2.2.2.1
  have c7719 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.1
  have c7720 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.1
  have c7721 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.1
  have e_198_14 := arithEq_of_rows h (row := 198) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_14
  simp only [← c7716, ← c7717, ← c7718] at e_198_14
  have e_199_0 := arithEq_of_rows h (row := 199) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_0
  simp only [← c7719, ← c7720, ← c7721] at e_199_0
  have hr := e_199_0
  simp only [e_198_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1458 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 11) = bselect (a (.virt 416)) (a (.wire 194 19)) (a (.virt 198)) := by
  have c7722 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7723 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7724 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7725 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7726 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7727 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_1 := arithEq_of_rows h (row := 199) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_1
  simp only [← c7722, ← c7723, ← c7724] at e_199_1
  have e_199_2 := arithEq_of_rows h (row := 199) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_2
  simp only [← c7725, ← c7726, ← c7727] at e_199_2
  have hr := e_199_2
  simp only [e_199_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1459 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 19) = bselect (a (.virt 416)) (a (.virt 195)) (a (.virt 199)) := by
  have c7728 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7729 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7730 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7731 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7732 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7733 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_3 := arithEq_of_rows h (row := 199) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_3
  simp only [← c7728, ← c7729, ← c7730] at e_199_3
  have e_199_4 := arithEq_of_rows h (row := 199) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_4
  simp only [← c7731, ← c7732, ← c7733] at e_199_4
  have hr := e_199_4
  simp only [e_199_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1460 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 27) = bselect (a (.virt 418)) (a (.wire 189 55)) (a (.wire 199 19)) := by
  have c7734 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7735 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7736 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7737 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7738 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7739 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_5 := arithEq_of_rows h (row := 199) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_5
  simp only [← c7734, ← c7735, ← c7736] at e_199_5
  have e_199_6 := arithEq_of_rows h (row := 199) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_6
  simp only [← c7737, ← c7738, ← c7739] at e_199_6
  have hr := e_199_6
  simp only [e_199_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1461 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 35) = bselect (a (.virt 416)) (a (.virt 196)) (a (.virt 200)) := by
  have c7740 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7741 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7742 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7743 := (leafCircuit_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7744 := (leafCircuit_copies242 a h).1
  have c7745 := (leafCircuit_copies242 a h).2.1
  have e_199_7 := arithEq_of_rows h (row := 199) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_7
  simp only [← c7740, ← c7741, ← c7742] at e_199_7
  have e_199_8 := arithEq_of_rows h (row := 199) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_8
  simp only [← c7743, ← c7744, ← c7745] at e_199_8
  have hr := e_199_8
  simp only [e_199_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1462 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 43) = bselect (a (.virt 418)) (a (.wire 194 3)) (a (.wire 199 35)) := by
  have c7746 := (leafCircuit_copies242 a h).2.2.1
  have c7747 := (leafCircuit_copies242 a h).2.2.2.1
  have c7748 := (leafCircuit_copies242 a h).2.2.2.2.1
  have c7749 := (leafCircuit_copies242 a h).2.2.2.2.2.1
  have c7750 := (leafCircuit_copies242 a h).2.2.2.2.2.2.1
  have c7751 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.1
  have e_199_9 := arithEq_of_rows h (row := 199) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_9
  simp only [← c7746, ← c7747, ← c7748] at e_199_9
  have e_199_10 := arithEq_of_rows h (row := 199) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_10
  simp only [← c7749, ← c7750, ← c7751] at e_199_10
  have hr := e_199_10
  simp only [e_199_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1463 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 51) = bselect (a (.virt 416)) (a (.virt 197)) (a (.virt 201)) := by
  have c7752 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.1
  have c7753 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.1
  have c7754 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7755 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7756 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7757 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_11 := arithEq_of_rows h (row := 199) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_11
  simp only [← c7752, ← c7753, ← c7754] at e_199_11
  have e_199_12 := arithEq_of_rows h (row := 199) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_12
  simp only [← c7755, ← c7756, ← c7757] at e_199_12
  have hr := e_199_12
  simp only [e_199_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1464 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 199 59) = bselect (a (.virt 418)) (a (.wire 194 11)) (a (.wire 199 51)) := by
  have c7758 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7759 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7760 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7761 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7762 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7763 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_13 := arithEq_of_rows h (row := 199) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_13
  simp only [← c7758, ← c7759, ← c7760] at e_199_13
  have e_199_14 := arithEq_of_rows h (row := 199) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_14
  simp only [← c7761, ← c7762, ← c7763] at e_199_14
  have hr := e_199_14
  simp only [e_199_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1465 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 7) = bselect (a (.virt 416)) (a (.virt 198)) (a (.virt 202)) := by
  have c7764 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7765 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7766 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7767 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7768 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7769 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_200_0 := arithEq_of_rows h (row := 200) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_0
  simp only [← c7764, ← c7765, ← c7766] at e_200_0
  have e_200_1 := arithEq_of_rows h (row := 200) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_1
  simp only [← c7767, ← c7768, ← c7769] at e_200_1
  have hr := e_200_1
  simp only [e_200_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1466 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 15) = bselect (a (.virt 418)) (a (.wire 194 19)) (a (.wire 200 7)) := by
  have c7770 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7771 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7772 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7773 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7774 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7775 := (leafCircuit_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_200_2 := arithEq_of_rows h (row := 200) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_2
  simp only [← c7770, ← c7771, ← c7772] at e_200_2
  have e_200_3 := arithEq_of_rows h (row := 200) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_3
  simp only [← c7773, ← c7774, ← c7775] at e_200_3
  have hr := e_200_3
  simp only [e_200_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1467 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) := by
  have c7776 := (leafCircuit_copies243 a h).1
  have c7777 := (leafCircuit_copies243 a h).2.1
  have c7778 := (leafCircuit_copies243 a h).2.2.1
  have c7779 := (leafCircuit_copies243 a h).2.2.2.1
  have c7780 := (leafCircuit_copies243 a h).2.2.2.2.1
  have c7781 := (leafCircuit_copies243 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_5 := arithEq_of_rows h (row := 164) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_5
  simp only [← c7776, ← c7777, ← c7778] at e_164_5
  have e_191_9 := arithEq_of_rows h (row := 191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_9
  simp only [← c7779, ← c7780, k3, ← c7781] at e_191_9
  have hr := e_191_9
  simp only [e_164_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1468 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 23) = bselect (a (.wire 191 39)) (a (.virt 199)) (a (.virt 203)) := by
  have c7782 := (leafCircuit_copies243 a h).2.2.2.2.2.2.1
  have c7783 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.1
  have c7784 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.1
  have c7785 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.1
  have c7786 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7787 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_200_4 := arithEq_of_rows h (row := 200) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_4
  simp only [← c7782, ← c7783, ← c7784] at e_200_4
  have e_200_5 := arithEq_of_rows h (row := 200) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_5
  simp only [← c7785, ← c7786, ← c7787] at e_200_5
  have hr := e_200_5
  simp only [e_200_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1469 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 31) = bselect (a (.virt 420)) (a (.wire 189 55)) (a (.wire 200 23)) := by
  have c7788 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7789 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7790 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7791 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7792 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7793 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_200_6 := arithEq_of_rows h (row := 200) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_6
  simp only [← c7788, ← c7789, ← c7790] at e_200_6
  have e_200_7 := arithEq_of_rows h (row := 200) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_7
  simp only [← c7791, ← c7792, ← c7793] at e_200_7
  have hr := e_200_7
  simp only [e_200_6] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1470 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) := by
  have c7776 := (leafCircuit_copies243 a h).1
  have c7777 := (leafCircuit_copies243 a h).2.1
  have c7778 := (leafCircuit_copies243 a h).2.2.1
  have c7779 := (leafCircuit_copies243 a h).2.2.2.1
  have c7780 := (leafCircuit_copies243 a h).2.2.2.2.1
  have c7781 := (leafCircuit_copies243 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_5 := arithEq_of_rows h (row := 164) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_5
  simp only [← c7776, ← c7777, ← c7778] at e_164_5
  have e_191_9 := arithEq_of_rows h (row := 191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_9
  simp only [← c7779, ← c7780, k3, ← c7781] at e_191_9
  have hr := e_191_9
  simp only [e_164_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1471 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 39) = bselect (a (.wire 191 39)) (a (.virt 200)) (a (.virt 204)) := by
  have c7794 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7795 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7796 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7797 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7798 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7799 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_200_8 := arithEq_of_rows h (row := 200) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_8
  simp only [← c7794, ← c7795, ← c7796] at e_200_8
  have e_200_9 := arithEq_of_rows h (row := 200) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_9
  simp only [← c7797, ← c7798, ← c7799] at e_200_9
  have hr := e_200_9
  simp only [e_200_8] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1472 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 47) = bselect (a (.virt 420)) (a (.wire 194 3)) (a (.wire 200 39)) := by
  have c7800 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7801 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7802 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7803 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7804 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7805 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_200_10 := arithEq_of_rows h (row := 200) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_10
  simp only [← c7800, ← c7801, ← c7802] at e_200_10
  have e_200_11 := arithEq_of_rows h (row := 200) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_11
  simp only [← c7803, ← c7804, ← c7805] at e_200_11
  have hr := e_200_11
  simp only [e_200_10] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1473 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) := by
  have c7776 := (leafCircuit_copies243 a h).1
  have c7777 := (leafCircuit_copies243 a h).2.1
  have c7778 := (leafCircuit_copies243 a h).2.2.1
  have c7779 := (leafCircuit_copies243 a h).2.2.2.1
  have c7780 := (leafCircuit_copies243 a h).2.2.2.2.1
  have c7781 := (leafCircuit_copies243 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_5 := arithEq_of_rows h (row := 164) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_5
  simp only [← c7776, ← c7777, ← c7778] at e_164_5
  have e_191_9 := arithEq_of_rows h (row := 191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_9
  simp only [← c7779, ← c7780, k3, ← c7781] at e_191_9
  have hr := e_191_9
  simp only [e_164_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1474 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 200 55) = bselect (a (.wire 191 39)) (a (.virt 201)) (a (.virt 205)) := by
  have c7806 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7807 := (leafCircuit_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7808 := (leafCircuit_copies244 a h).1
  have c7809 := (leafCircuit_copies244 a h).2.1
  have c7810 := (leafCircuit_copies244 a h).2.2.1
  have c7811 := (leafCircuit_copies244 a h).2.2.2.1
  have e_200_12 := arithEq_of_rows h (row := 200) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_12
  simp only [← c7806, ← c7807, ← c7808] at e_200_12
  have e_200_13 := arithEq_of_rows h (row := 200) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_13
  simp only [← c7809, ← c7810, ← c7811] at e_200_13
  have hr := e_200_13
  simp only [e_200_12] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1475 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 3) = bselect (a (.virt 420)) (a (.wire 194 11)) (a (.wire 200 55)) := by
  have c7812 := (leafCircuit_copies244 a h).2.2.2.2.1
  have c7813 := (leafCircuit_copies244 a h).2.2.2.2.2.1
  have c7814 := (leafCircuit_copies244 a h).2.2.2.2.2.2.1
  have c7815 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.1
  have c7816 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.1
  have c7817 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.1
  have e_200_14 := arithEq_of_rows h (row := 200) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_14
  simp only [← c7812, ← c7813, ← c7814] at e_200_14
  have e_201_0 := arithEq_of_rows h (row := 201) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_0
  simp only [← c7815, ← c7816, ← c7817] at e_201_0
  have hr := e_201_0
  simp only [e_200_14] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1476 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) := by
  have c7776 := (leafCircuit_copies243 a h).1
  have c7777 := (leafCircuit_copies243 a h).2.1
  have c7778 := (leafCircuit_copies243 a h).2.2.1
  have c7779 := (leafCircuit_copies243 a h).2.2.2.1
  have c7780 := (leafCircuit_copies243 a h).2.2.2.2.1
  have c7781 := (leafCircuit_copies243 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_164_5 := arithEq_of_rows h (row := 164) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_5
  simp only [← c7776, ← c7777, ← c7778] at e_164_5
  have e_191_9 := arithEq_of_rows h (row := 191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_9
  simp only [← c7779, ← c7780, k3, ← c7781] at e_191_9
  have hr := e_191_9
  simp only [e_164_5] at hr
  simp only [bor]
  linear_combination hr

theorem leafCircuit_f1477 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 11) = bselect (a (.wire 191 39)) (a (.virt 202)) (a (.virt 206)) := by
  have c7818 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7819 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7820 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7821 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7822 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7823 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_1 := arithEq_of_rows h (row := 201) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_1
  simp only [← c7818, ← c7819, ← c7820] at e_201_1
  have e_201_2 := arithEq_of_rows h (row := 201) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_2
  simp only [← c7821, ← c7822, ← c7823] at e_201_2
  have hr := e_201_2
  simp only [e_201_1] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1478 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 19) = bselect (a (.virt 420)) (a (.wire 194 19)) (a (.wire 201 11)) := by
  have c7824 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7825 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7826 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7827 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7828 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7829 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_3 := arithEq_of_rows h (row := 201) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_3
  simp only [← c7824, ← c7825, ← c7826] at e_201_3
  have e_201_4 := arithEq_of_rows h (row := 201) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_4
  simp only [← c7827, ← c7828, ← c7829] at e_201_4
  have hr := e_201_4
  simp only [e_201_3] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1479 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 27) = bselect (a (.virt 422)) (a (.wire 189 55)) (a (.virt 203)) := by
  have c7830 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7831 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7832 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7833 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7834 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7835 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_5 := arithEq_of_rows h (row := 201) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_5
  simp only [← c7830, ← c7831, ← c7832] at e_201_5
  have e_201_6 := arithEq_of_rows h (row := 201) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_6
  simp only [← c7833, ← c7834, ← c7835] at e_201_6
  have hr := e_201_6
  simp only [e_201_5] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1480 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 35) = bselect (a (.virt 422)) (a (.wire 194 3)) (a (.virt 204)) := by
  have c7836 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7837 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7838 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7839 := (leafCircuit_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7840 := (leafCircuit_copies245 a h).1
  have c7841 := (leafCircuit_copies245 a h).2.1
  have e_201_7 := arithEq_of_rows h (row := 201) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_7
  simp only [← c7836, ← c7837, ← c7838] at e_201_7
  have e_201_8 := arithEq_of_rows h (row := 201) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_8
  simp only [← c7839, ← c7840, ← c7841] at e_201_8
  have hr := e_201_8
  simp only [e_201_7] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1481 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 43) = bselect (a (.virt 422)) (a (.wire 194 11)) (a (.virt 205)) := by
  have c7842 := (leafCircuit_copies245 a h).2.2.1
  have c7843 := (leafCircuit_copies245 a h).2.2.2.1
  have c7844 := (leafCircuit_copies245 a h).2.2.2.2.1
  have c7845 := (leafCircuit_copies245 a h).2.2.2.2.2.1
  have c7846 := (leafCircuit_copies245 a h).2.2.2.2.2.2.1
  have c7847 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.1
  have e_201_9 := arithEq_of_rows h (row := 201) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_9
  simp only [← c7842, ← c7843, ← c7844] at e_201_9
  have e_201_10 := arithEq_of_rows h (row := 201) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_10
  simp only [← c7845, ← c7846, ← c7847] at e_201_10
  have hr := e_201_10
  simp only [e_201_9] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1482 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 51) = bselect (a (.virt 422)) (a (.wire 194 19)) (a (.virt 206)) := by
  have c7848 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.1
  have c7849 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.1
  have c7850 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7851 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7852 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7853 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_11 := arithEq_of_rows h (row := 201) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_11
  simp only [← c7848, ← c7849, ← c7850] at e_201_11
  have e_201_12 := arithEq_of_rows h (row := 201) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_12
  simp only [← c7851, ← c7852, ← c7853] at e_201_12
  have hr := e_201_12
  simp only [e_201_11] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1483 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 205 12) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 0 ∧ a (.wire 205 13) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 1 ∧ a (.wire 205 14) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 2 ∧ a (.wire 205 15) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 3) := by
  have c7854 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7855 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7856 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7857 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7858 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7859 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7860 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7861 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7862 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7863 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7864 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7865 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7866 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7867 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7868 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7869 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7870 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7871 := (leafCircuit_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7872 := (leafCircuit_copies246 a h).1
  have c7873 := (leafCircuit_copies246 a h).2.1
  have c7874 := (leafCircuit_copies246 a h).2.2.1
  have c7875 := (leafCircuit_copies246 a h).2.2.2.1
  have c7876 := (leafCircuit_copies246 a h).2.2.2.2.1
  have c7877 := (leafCircuit_copies246 a h).2.2.2.2.2.1
  have c7878 := (leafCircuit_copies246 a h).2.2.2.2.2.2.1
  have c7879 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.1
  have c7880 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.1
  have c7881 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.1
  have c7882 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7883 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7884 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7885 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7886 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7887 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7888 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7889 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7890 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7891 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7892 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7893 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7894 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7895 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7896 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7897 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7898 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7899 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7900 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7901 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7902 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7903 := (leafCircuit_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7904 := (leafCircuit_copies247 a h).1
  have c7905 := (leafCircuit_copies247 a h).2.1
  have c7906 := (leafCircuit_copies247 a h).2.2.1
  have c7907 := (leafCircuit_copies247 a h).2.2.2.1
  have c7908 := (leafCircuit_copies247 a h).2.2.2.2.1
  have c7909 := (leafCircuit_copies247 a h).2.2.2.2.2.1
  have c7910 := (leafCircuit_copies247 a h).2.2.2.2.2.2.1
  have c7911 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.1
  have c7912 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.1
  have c7913 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.1
  have c7914 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7915 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7916 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_191_10 := arithEq_of_rows h (row := 191) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_10
  simp only [← c7866, ← c7867, k3, ← c7868] at e_191_10
  have e_191_11 := arithEq_of_rows h (row := 191) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_11
  simp only [← c7869, ← c7870, k3, ← c7871] at e_191_11
  have e_191_12 := arithEq_of_rows h (row := 191) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_12
  simp only [← c7872, ← c7873, k3, ← c7874] at e_191_12
  have e_191_13 := arithEq_of_rows h (row := 191) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_13
  simp only [← c7875, ← c7876, k3, ← c7877] at e_191_13
  have e_191_14 := arithEq_of_rows h (row := 191) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_14
  simp only [← c7878, ← c7879, k3, ← c7880] at e_191_14
  have e_203_0 := arithEq_of_rows h (row := 203) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_0
  simp only [← c7881, ← c7882, k3, ← c7883] at e_203_0
  have e_203_1 := arithEq_of_rows h (row := 203) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_1
  simp only [← c7884, ← c7885, k3, ← c7886] at e_203_1
  have e_203_2 := arithEq_of_rows h (row := 203) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_2
  simp only [← c7887, ← c7888, k3, ← c7889] at e_203_2
  have e_203_3 := arithEq_of_rows h (row := 203) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_3
  simp only [← c7902, ← c7903, k3, ← c7904, k3] at e_203_3
  have hin0 : poseidon2In a 202 = addBlock (fun _ => 0) [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15)] :=
    poseidon2In_first c7854.symm c7855.symm c7856.symm c7857.symm c7858.symm c7859.symm c7860.symm c7861.symm (c7862.symm.trans k0) (c7863.symm.trans k0) (c7864.symm.trans k0) (c7865.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 202) rfl rfl hin0
  have hin1 : poseidon2In a 204 = addBlock (poseidon2Out a 202) [a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] :=
    poseidon2In_chain (by linear_combination -c7890 + e_191_10) (by linear_combination -c7891 + e_191_11) (by linear_combination -c7892 + e_191_12) (by linear_combination -c7893 + e_191_13) (by linear_combination -c7894 + e_191_14) (by linear_combination -c7895 + e_203_0) (by linear_combination -c7896 + e_203_1) (by linear_combination -c7897 + e_203_2) c7898.symm c7899.symm c7900.symm c7901.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 204) rfl rfl hin1
  have hin2 : poseidon2In a 205 = addBlock (poseidon2Out a 204) [1, 0, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c7905 + e_203_3) (c7906.symm.trans (add_zero _).symm) (c7907.symm.trans (add_zero _).symm) (c7908.symm.trans (add_zero _).symm) (c7909.symm.trans (add_zero _).symm) (c7910.symm.trans (add_zero _).symm) (c7911.symm.trans (add_zero _).symm) (c7912.symm.trans (add_zero _).symm) c7913.symm c7914.symm c7915.symm c7916.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 205) rfl rfl hin2
  have hpad : pad10 [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] =
      [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15)] ++ ([a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] ++ ([1, 0, 0, 0, 0, 0, 0, 0] ++ [])) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1484 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 201 59) = bselect (a (.wire 195 5)) (a (.wire 205 12)) (a (.wire 189 55)) := by
  have c7917 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7918 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7919 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7920 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7921 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7922 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_13 := arithEq_of_rows h (row := 201) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_13
  simp only [← c7917, ← c7918, ← c7919] at e_201_13
  have e_201_14 := arithEq_of_rows h (row := 201) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_14
  simp only [← c7920, ← c7921, ← c7922] at e_201_14
  have hr := e_201_14
  simp only [e_201_13] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1485 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 7) = bselect (a (.wire 195 5)) (a (.wire 205 13)) (a (.wire 194 3)) := by
  have c7923 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7924 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7925 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7926 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7927 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7928 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_206_0 := arithEq_of_rows h (row := 206) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_0
  simp only [← c7923, ← c7924, ← c7925] at e_206_0
  have e_206_1 := arithEq_of_rows h (row := 206) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_1
  simp only [← c7926, ← c7927, ← c7928] at e_206_1
  have hr := e_206_1
  simp only [e_206_0] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1486 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 15) = bselect (a (.wire 195 5)) (a (.wire 205 14)) (a (.wire 194 11)) := by
  have c7929 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7930 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7931 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7932 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7933 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7934 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_206_2 := arithEq_of_rows h (row := 206) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_2
  simp only [← c7929, ← c7930, ← c7931] at e_206_2
  have e_206_3 := arithEq_of_rows h (row := 206) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_3
  simp only [← c7932, ← c7933, ← c7934] at e_206_3
  have hr := e_206_3
  simp only [e_206_2] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1487 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 23) = bselect (a (.wire 195 5)) (a (.wire 205 15)) (a (.wire 194 19)) := by
  have c7935 := (leafCircuit_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7936 := (leafCircuit_copies248 a h).1
  have c7937 := (leafCircuit_copies248 a h).2.1
  have c7938 := (leafCircuit_copies248 a h).2.2.1
  have c7939 := (leafCircuit_copies248 a h).2.2.2.1
  have c7940 := (leafCircuit_copies248 a h).2.2.2.2.1
  have e_206_4 := arithEq_of_rows h (row := 206) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_4
  simp only [← c7935, ← c7936, ← c7937] at e_206_4
  have e_206_5 := arithEq_of_rows h (row := 206) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_5
  simp only [← c7938, ← c7939, ← c7940] at e_206_5
  have hr := e_206_5
  simp only [e_206_4] at hr
  simp only [bselect]
  linear_combination hr

theorem leafCircuit_f1488 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 27) = a (.wire 201 59) - a (.virt 8) := by
  have c7941 := (leafCircuit_copies248 a h).2.2.2.2.2.1
  have c7942 := (leafCircuit_copies248 a h).2.2.2.2.2.2.1
  have c7943 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_206_6 := arithEq_of_rows h (row := 206) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_6
  simp only [← c7941, ← c7942, k3, ← c7943] at e_206_6
  have hr := e_206_6
  linear_combination hr

theorem leafCircuit_f1489 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 43) = a (.wire 206 27) * a (.virt 13) := by
  have c7944 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.1
  have c7945 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.1
  have c7946 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_196_10 := arithEq_of_rows h (row := 196) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_10
  simp only [← c7944, ← c7945, ← c7946] at e_196_10
  have hr := e_196_10
  linear_combination hr

theorem leafCircuit_f1490 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 43) = a (.virt 14) := by
  have c7947 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c7947

theorem leafCircuit_f1491 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 31) = a (.wire 206 7) - a (.virt 9) := by
  have c7948 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7949 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7950 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_206_7 := arithEq_of_rows h (row := 206) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_7
  simp only [← c7948, ← c7949, k3, ← c7950] at e_206_7
  have hr := e_206_7
  linear_combination hr

theorem leafCircuit_f1492 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 47) = a (.wire 206 31) * a (.virt 13) := by
  have c7951 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7952 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7953 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_196_11 := arithEq_of_rows h (row := 196) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_11
  simp only [← c7951, ← c7952, ← c7953] at e_196_11
  have hr := e_196_11
  linear_combination hr

theorem leafCircuit_f1493 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 47) = a (.virt 14) := by
  have c7954 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c7954

theorem leafCircuit_f1494 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 35) = a (.wire 206 15) - a (.virt 10) := by
  have c7955 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7956 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7957 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_206_8 := arithEq_of_rows h (row := 206) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_8
  simp only [← c7955, ← c7956, k3, ← c7957] at e_206_8
  have hr := e_206_8
  linear_combination hr

theorem leafCircuit_f1495 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 51) = a (.wire 206 35) * a (.virt 13) := by
  have c7958 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7959 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7960 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_196_12 := arithEq_of_rows h (row := 196) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_12
  simp only [← c7958, ← c7959, ← c7960] at e_196_12
  have hr := e_196_12
  linear_combination hr

theorem leafCircuit_f1496 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 51) = a (.virt 14) := by
  have c7961 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c7961

theorem leafCircuit_f1497 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 206 39) = a (.wire 206 23) - a (.virt 11) := by
  have c7962 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7963 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7964 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_206_9 := arithEq_of_rows h (row := 206) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_9
  simp only [← c7962, ← c7963, k3, ← c7964] at e_206_9
  have hr := e_206_9
  linear_combination hr

theorem leafCircuit_f1498 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 55) = a (.wire 206 39) * a (.virt 13) := by
  have c7965 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7966 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7967 := (leafCircuit_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_196_13 := arithEq_of_rows h (row := 196) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_13
  simp only [← c7965, ← c7966, ← c7967] at e_196_13
  have hr := e_196_13
  linear_combination hr

theorem leafCircuit_f1499 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 196 55) = a (.virt 14) := by
  have c7968 := (leafCircuit_copies249 a h).1
  exact c7968

theorem leafCircuit_f1500 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    rangeCheck (a (.virt 249)) 32 := by
  have c7969 := (leafCircuit_copies249 a h).2.1
  have c7970 := (leafCircuit_copies249 a h).2.2.1
  have c7971 := (leafCircuit_copies249 a h).2.2.2.1
  have c7972 := (leafCircuit_copies249 a h).2.2.2.2.1
  have c7973 := (leafCircuit_copies249 a h).2.2.2.2.2.1
  have c7974 := (leafCircuit_copies249 a h).2.2.2.2.2.2.1
  have c7975 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.1
  have c7976 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.1
  have c7977 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.1
  have c7978 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7979 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7980 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7981 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7982 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7983 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7984 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7985 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7986 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7987 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7988 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7989 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7990 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7991 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7992 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7993 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7994 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7995 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7996 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hr := rangeCheck_of_row h (row := 207) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c7969.trans k0
    · exact c7970.trans k0
    · exact c7971.trans k0
    · exact c7972.trans k0
    · exact c7973.trans k0
    · exact c7974.trans k0
    · exact c7975.trans k0
    · exact c7976.trans k0
    · exact c7977.trans k0
    · exact c7978.trans k0
    · exact c7979.trans k0
    · exact c7980.trans k0
    · exact c7981.trans k0
    · exact c7982.trans k0
    · exact c7983.trans k0
    · exact c7984.trans k0
    · exact c7985.trans k0
    · exact c7986.trans k0
    · exact c7987.trans k0
    · exact c7988.trans k0
    · exact c7989.trans k0
    · exact c7990.trans k0
    · exact c7991.trans k0
    · exact c7992.trans k0
    · exact c7993.trans k0
    · exact c7994.trans k0
    · exact c7995.trans k0
    )
  rwa [c7996] at hr

theorem leafCircuit_f1501 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 227) = a (.virt 237) := by
  have c7997 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c7997

theorem leafCircuit_f1502 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 228) = a (.virt 238) := by
  have c7998 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c7998

theorem leafCircuit_f1503 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 229) = a (.virt 239) := by
  have c7999 := (leafCircuit_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c7999

theorem leafCircuit_f1504 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 230) = a (.virt 240) := by
  have c8000 := (leafCircuit_copies250 a h).1
  exact c8000

theorem leafCircuit_f1505 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 231) = a (.virt 5) := by
  have c8001 := (leafCircuit_copies250 a h).2.1
  exact c8001

theorem leafCircuit_f1506 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 232) = a (.virt 6) := by
  have c8002 := (leafCircuit_copies250 a h).2.2.1
  exact c8002

theorem leafCircuit_f1507 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 233) = a (.virt 1) := by
  have c8003 := (leafCircuit_copies250 a h).2.2.2.1
  exact c8003

theorem leafCircuit_f1508 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 234) = a (.virt 2) := by
  have c8004 := (leafCircuit_copies250 a h).2.2.2.2.1
  exact c8004

theorem leafCircuit_f1509 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 235) = a (.virt 3) := by
  have c8005 := (leafCircuit_copies250 a h).2.2.2.2.2.1
  exact c8005

theorem leafCircuit_f1510 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 236) = a (.virt 4) := by
  have c8006 := (leafCircuit_copies250 a h).2.2.2.2.2.2.1
  exact c8006

theorem leafCircuit_f1511 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 241)) (a (.virt 14)) (a (.virt 424)) (a (.virt 425)) := by
  have c8007 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.1
  have c8008 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.1
  have c8009 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.1
  have c8010 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8011 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8012 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8013 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8014 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8015 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8016 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8017 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8018 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8019 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8020 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_196_14 := arithEq_of_rows h (row := 196) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_14
  simp only [← c8010, ← c8011, ← c8012] at e_196_14
  have e_206_10 := arithEq_of_rows h (row := 206) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_10
  simp only [← c8007, k3, ← c8008, k3, ← c8009] at e_206_10
  have e_206_11 := arithEq_of_rows h (row := 206) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_11
  simp only [← c8016, ← c8017, k3, ← c8018] at e_206_11
  have e_208_0 := arithEq_of_rows h (row := 208) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_0
  simp only [← c8013, ← c8014, ← c8015] at e_208_0
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_196_14
    linear_combination c8019.trans k0 - hc
  · have hc := e_206_11
    simp only [e_206_10, e_208_0] at hc
    linear_combination c8020.trans k0 - hc

theorem leafCircuit_f1512 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 242)) (a (.virt 14)) (a (.virt 426)) (a (.virt 427)) := by
  have c8021 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8022 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8023 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8024 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8025 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8026 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8027 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8028 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8029 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8030 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8031 := (leafCircuit_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8032 := (leafCircuit_copies251 a h).1
  have c8033 := (leafCircuit_copies251 a h).2.1
  have c8034 := (leafCircuit_copies251 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_206_12 := arithEq_of_rows h (row := 206) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_12
  simp only [← c8021, k3, ← c8022, k3, ← c8023] at e_206_12
  have e_206_13 := arithEq_of_rows h (row := 206) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_13
  simp only [← c8030, ← c8031, k3, ← c8032] at e_206_13
  have e_208_1 := arithEq_of_rows h (row := 208) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_1
  simp only [← c8024, ← c8025, ← c8026] at e_208_1
  have e_208_2 := arithEq_of_rows h (row := 208) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_2
  simp only [← c8027, ← c8028, ← c8029] at e_208_2
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_208_1
    linear_combination c8033.trans k0 - hc
  · have hc := e_206_13
    simp only [e_206_12, e_208_2] at hc
    linear_combination c8034.trans k0 - hc

theorem leafCircuit_f1513 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 243)) (a (.virt 14)) (a (.virt 428)) (a (.virt 429)) := by
  have c8035 := (leafCircuit_copies251 a h).2.2.2.1
  have c8036 := (leafCircuit_copies251 a h).2.2.2.2.1
  have c8037 := (leafCircuit_copies251 a h).2.2.2.2.2.1
  have c8038 := (leafCircuit_copies251 a h).2.2.2.2.2.2.1
  have c8039 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.1
  have c8040 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.1
  have c8041 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.1
  have c8042 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8043 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8044 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8045 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8046 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8047 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8048 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_206_14 := arithEq_of_rows h (row := 206) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_14
  simp only [← c8035, k3, ← c8036, k3, ← c8037] at e_206_14
  have e_208_3 := arithEq_of_rows h (row := 208) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_3
  simp only [← c8038, ← c8039, ← c8040] at e_208_3
  have e_208_4 := arithEq_of_rows h (row := 208) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_4
  simp only [← c8041, ← c8042, ← c8043] at e_208_4
  have e_209_0 := arithEq_of_rows h (row := 209) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_0
  simp only [← c8044, ← c8045, k3, ← c8046] at e_209_0
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_208_3
    linear_combination c8047.trans k0 - hc
  · have hc := e_209_0
    simp only [e_206_14, e_208_4] at hc
    linear_combination c8048.trans k0 - hc

theorem leafCircuit_f1514 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    IsEqual (a (.virt 244)) (a (.virt 14)) (a (.virt 430)) (a (.virt 431)) := by
  have c8049 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8050 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8051 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8052 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8053 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8054 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8055 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8056 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8057 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8058 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8059 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8060 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8061 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8062 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_208_5 := arithEq_of_rows h (row := 208) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_5
  simp only [← c8052, ← c8053, ← c8054] at e_208_5
  have e_208_6 := arithEq_of_rows h (row := 208) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_6
  simp only [← c8055, ← c8056, ← c8057] at e_208_6
  have e_209_1 := arithEq_of_rows h (row := 209) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_1
  simp only [← c8049, k3, ← c8050, k3, ← c8051] at e_209_1
  have e_209_2 := arithEq_of_rows h (row := 209) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_2
  simp only [← c8058, ← c8059, k3, ← c8060] at e_209_2
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_208_5
    linear_combination c8061.trans k0 - hc
  · have hc := e_209_2
    simp only [e_209_1, e_208_6] at hc
    linear_combination c8062.trans k0 - hc

theorem leafCircuit_f1515 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 31) = band (a (.virt 424)) (a (.virt 426)) := by
  have c8063 := (leafCircuit_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8064 := (leafCircuit_copies252 a h).1
  have c8065 := (leafCircuit_copies252 a h).2.1
  have e_208_7 := arithEq_of_rows h (row := 208) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_7
  simp only [← c8063, ← c8064, ← c8065] at e_208_7
  have hr := e_208_7
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1516 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 35) = band (a (.virt 428)) (a (.virt 430)) := by
  have c8066 := (leafCircuit_copies252 a h).2.2.1
  have c8067 := (leafCircuit_copies252 a h).2.2.2.1
  have c8068 := (leafCircuit_copies252 a h).2.2.2.2.1
  have e_208_8 := arithEq_of_rows h (row := 208) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_8
  simp only [← c8066, ← c8067, ← c8068] at e_208_8
  have hr := e_208_8
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1517 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 39) = band (a (.wire 208 31)) (a (.wire 208 35)) := by
  have c8069 := (leafCircuit_copies252 a h).2.2.2.2.2.1
  have c8070 := (leafCircuit_copies252 a h).2.2.2.2.2.2.1
  have c8071 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.1
  have e_208_9 := arithEq_of_rows h (row := 208) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_9
  simp only [← c8069, ← c8070, ← c8071] at e_208_9
  have hr := e_208_9
  simp only [band]
  linear_combination hr

theorem leafCircuit_f1518 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 15) = a (.virt 292) - a (.wire 208 39) := by
  have c8072 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.1
  have c8073 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.1
  have c8074 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_3 := arithEq_of_rows h (row := 209) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_3
  simp only [← c8072, k3, ← c8073, k3, ← c8074] at e_209_3
  have hr := e_209_3
  simp only [k3]
  linear_combination hr

theorem leafCircuit_f1519 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 43) = a (.wire 208 39) * a (.virt 7) := by
  have c8075 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8076 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8077 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_208_10 := arithEq_of_rows h (row := 208) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_10
  simp only [← c8075, ← c8076, ← c8077] at e_208_10
  have hr := e_208_10
  linear_combination hr

theorem leafCircuit_f1520 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 43) = a (.virt 14) := by
  have c8078 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8078

theorem leafCircuit_f1521 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.virt 13) = a (.wire 209 15) := by
  have c8079 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8079

theorem leafCircuit_f1522 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 211 12) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 0 ∧ a (.wire 211 13) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 1 ∧ a (.wire 211 14) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 2 ∧ a (.wire 211 15) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 3) := by
  have c8080 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8081 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8082 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8083 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8084 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8085 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8086 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8087 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8088 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8089 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8090 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8091 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8092 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8093 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8094 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8095 := (leafCircuit_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8096 := (leafCircuit_copies253 a h).1
  have c8097 := (leafCircuit_copies253 a h).2.1
  have c8098 := (leafCircuit_copies253 a h).2.2.1
  have c8099 := (leafCircuit_copies253 a h).2.2.2.1
  have c8100 := (leafCircuit_copies253 a h).2.2.2.2.1
  have c8101 := (leafCircuit_copies253 a h).2.2.2.2.2.1
  have c8102 := (leafCircuit_copies253 a h).2.2.2.2.2.2.1
  have c8103 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.1
  have c8104 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.1
  have c8105 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.1
  have c8106 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8107 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8108 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8109 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_203_4 := arithEq_of_rows h (row := 203) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_4
  simp only [← c8092, ← c8093, k3, ← c8094] at e_203_4
  have e_203_5 := arithEq_of_rows h (row := 203) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_5
  simp only [← c8095, ← c8096, k3, ← c8097, k3] at e_203_5
  have hin0 : poseidon2In a 210 = addBlock (fun _ => 0) [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231)] :=
    poseidon2In_first c8080.symm c8081.symm c8082.symm c8083.symm c8084.symm c8085.symm c8086.symm c8087.symm (c8088.symm.trans k0) (c8089.symm.trans k0) (c8090.symm.trans k0) (c8091.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 210) rfl rfl hin0
  have hin1 : poseidon2In a 211 = addBlock (poseidon2Out a 210) [a (.virt 232), 1, 0, 0, 0, 0, 0, 0] :=
    poseidon2In_chain (by linear_combination -c8098 + e_203_4) (by linear_combination -c8099 + e_203_5) (c8100.symm.trans (add_zero _).symm) (c8101.symm.trans (add_zero _).symm) (c8102.symm.trans (add_zero _).symm) (c8103.symm.trans (add_zero _).symm) (c8104.symm.trans (add_zero _).symm) (c8105.symm.trans (add_zero _).symm) c8106.symm c8107.symm c8108.symm c8109.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 211) rfl rfl hin1
  have hpad : pad10 [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] =
      [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231)] ++ ([a (.virt 232), 1, 0, 0, 0, 0, 0, 0] ++ []) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1523 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 212 12) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 0 ∧ a (.wire 212 13) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 1 ∧ a (.wire 212 14) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 2 ∧ a (.wire 212 15) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 3) := by
  have c8110 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8111 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8112 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8113 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8114 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8115 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8116 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8117 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8118 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8119 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8120 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8121 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have hin0 : poseidon2In a 212 = addBlock (fun _ => 0) [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15), 1, 0, 0, 0] :=
    poseidon2In_first c8110.symm c8111.symm c8112.symm c8113.symm (c8114.symm.trans k3) (c8115.symm.trans k0) (c8116.symm.trans k0) (c8117.symm.trans k0) (c8118.symm.trans k0) (c8119.symm.trans k0) (c8120.symm.trans k0) (c8121.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 212) rfl rfl hin0
  have hpad : pad10 [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] =
      [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1524 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 19) = a (.virt 223) - a (.wire 212 12) := by
  have c8122 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8123 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8124 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_4 := arithEq_of_rows h (row := 209) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_4
  simp only [← c8122, ← c8123, k3, ← c8124] at e_209_4
  have hr := e_209_4
  linear_combination hr

theorem leafCircuit_f1525 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 47) = a (.wire 209 19) * a (.wire 209 15) := by
  have c8125 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8126 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8127 := (leafCircuit_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_208_11 := arithEq_of_rows h (row := 208) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_11
  simp only [← c8125, ← c8126, ← c8127] at e_208_11
  have hr := e_208_11
  linear_combination hr

theorem leafCircuit_f1526 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 47) = a (.virt 14) := by
  have c8128 := (leafCircuit_copies254 a h).1
  exact c8128

theorem leafCircuit_f1527 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 23) = a (.virt 224) - a (.wire 212 13) := by
  have c8129 := (leafCircuit_copies254 a h).2.1
  have c8130 := (leafCircuit_copies254 a h).2.2.1
  have c8131 := (leafCircuit_copies254 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_5 := arithEq_of_rows h (row := 209) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_5
  simp only [← c8129, ← c8130, k3, ← c8131] at e_209_5
  have hr := e_209_5
  linear_combination hr

theorem leafCircuit_f1528 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 51) = a (.wire 209 23) * a (.wire 209 15) := by
  have c8132 := (leafCircuit_copies254 a h).2.2.2.2.1
  have c8133 := (leafCircuit_copies254 a h).2.2.2.2.2.1
  have c8134 := (leafCircuit_copies254 a h).2.2.2.2.2.2.1
  have e_208_12 := arithEq_of_rows h (row := 208) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_12
  simp only [← c8132, ← c8133, ← c8134] at e_208_12
  have hr := e_208_12
  linear_combination hr

theorem leafCircuit_f1529 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 51) = a (.virt 14) := by
  have c8135 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.1
  exact c8135

theorem leafCircuit_f1530 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 27) = a (.virt 225) - a (.wire 212 14) := by
  have c8136 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.1
  have c8137 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.1
  have c8138 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_6 := arithEq_of_rows h (row := 209) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_6
  simp only [← c8136, ← c8137, k3, ← c8138] at e_209_6
  have hr := e_209_6
  linear_combination hr

theorem leafCircuit_f1531 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 55) = a (.wire 209 27) * a (.wire 209 15) := by
  have c8139 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8140 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8141 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_208_13 := arithEq_of_rows h (row := 208) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_13
  simp only [← c8139, ← c8140, ← c8141] at e_208_13
  have hr := e_208_13
  linear_combination hr

theorem leafCircuit_f1532 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 55) = a (.virt 14) := by
  have c8142 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8142

theorem leafCircuit_f1533 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 31) = a (.virt 226) - a (.wire 212 15) := by
  have c8143 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8144 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8145 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_7 := arithEq_of_rows h (row := 209) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_7
  simp only [← c8143, ← c8144, k3, ← c8145] at e_209_7
  have hr := e_209_7
  linear_combination hr

theorem leafCircuit_f1534 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 59) = a (.wire 209 31) * a (.wire 209 15) := by
  have c8146 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8147 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8148 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_208_14 := arithEq_of_rows h (row := 208) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_14
  simp only [← c8146, ← c8147, ← c8148] at e_208_14
  have hr := e_208_14
  linear_combination hr

theorem leafCircuit_f1535 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 208 59) = a (.virt 14) := by
  have c8149 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8149

theorem leafCircuit_f1536 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 220 12) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 0 ∧ a (.wire 220 13) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 1 ∧ a (.wire 220 14) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 2 ∧ a (.wire 220 15) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 3) := by
  have c8150 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8151 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8152 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8153 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8154 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8155 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8156 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8157 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8158 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8159 := (leafCircuit_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8160 := (leafCircuit_copies255 a h).1
  have c8161 := (leafCircuit_copies255 a h).2.1
  have c8162 := (leafCircuit_copies255 a h).2.2.1
  have c8163 := (leafCircuit_copies255 a h).2.2.2.1
  have c8164 := (leafCircuit_copies255 a h).2.2.2.2.1
  have c8165 := (leafCircuit_copies255 a h).2.2.2.2.2.1
  have c8166 := (leafCircuit_copies255 a h).2.2.2.2.2.2.1
  have c8167 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.1
  have c8168 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.1
  have c8169 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.1
  have c8170 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8171 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8172 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8173 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8174 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8175 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8176 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8177 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8178 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8179 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8180 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8181 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8182 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8183 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8184 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8185 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8186 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8187 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8188 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8189 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8190 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8191 := (leafCircuit_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8192 := (leafCircuit_copies256 a h).1
  have c8193 := (leafCircuit_copies256 a h).2.1
  have c8194 := (leafCircuit_copies256 a h).2.2.1
  have c8195 := (leafCircuit_copies256 a h).2.2.2.1
  have c8196 := (leafCircuit_copies256 a h).2.2.2.2.1
  have c8197 := (leafCircuit_copies256 a h).2.2.2.2.2.1
  have c8198 := (leafCircuit_copies256 a h).2.2.2.2.2.2.1
  have c8199 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.1
  have c8200 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.1
  have c8201 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.1
  have c8202 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8203 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8204 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8205 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8206 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8207 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8208 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8209 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8210 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8211 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8212 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8213 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8214 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8215 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8216 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8217 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8218 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8219 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8220 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8221 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8222 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8223 := (leafCircuit_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8224 := (leafCircuit_copies257 a h).1
  have c8225 := (leafCircuit_copies257 a h).2.1
  have c8226 := (leafCircuit_copies257 a h).2.2.1
  have c8227 := (leafCircuit_copies257 a h).2.2.2.1
  have c8228 := (leafCircuit_copies257 a h).2.2.2.2.1
  have c8229 := (leafCircuit_copies257 a h).2.2.2.2.2.1
  have c8230 := (leafCircuit_copies257 a h).2.2.2.2.2.2.1
  have c8231 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.1
  have c8232 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.1
  have c8233 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.1
  have c8234 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8235 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8236 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8237 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8238 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8239 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8240 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8241 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8242 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8243 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8244 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8245 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8246 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8247 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8248 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8249 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8250 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8251 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8252 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8253 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8254 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8255 := (leafCircuit_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8256 := (leafCircuit_copies258 a h).1
  have c8257 := (leafCircuit_copies258 a h).2.1
  have c8258 := (leafCircuit_copies258 a h).2.2.1
  have c8259 := (leafCircuit_copies258 a h).2.2.2.1
  have c8260 := (leafCircuit_copies258 a h).2.2.2.2.1
  have c8261 := (leafCircuit_copies258 a h).2.2.2.2.2.1
  have c8262 := (leafCircuit_copies258 a h).2.2.2.2.2.2.1
  have c8263 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.1
  have c8264 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.1
  have c8265 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.1
  have c8266 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8267 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8268 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8269 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8270 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8271 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8272 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8273 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8274 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8275 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8276 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8277 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8278 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8279 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8280 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8281 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8282 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8283 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8284 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8285 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8286 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8287 := (leafCircuit_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8288 := (leafCircuit_copies259 a h).1
  have c8289 := (leafCircuit_copies259 a h).2.1
  have c8290 := (leafCircuit_copies259 a h).2.2.1
  have c8291 := (leafCircuit_copies259 a h).2.2.2.1
  have c8292 := (leafCircuit_copies259 a h).2.2.2.2.1
  have c8293 := (leafCircuit_copies259 a h).2.2.2.2.2.1
  have c8294 := (leafCircuit_copies259 a h).2.2.2.2.2.2.1
  have c8295 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.1
  have c8296 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.1
  have c8297 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.1
  have c8298 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8299 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8300 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8301 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8302 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8303 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8304 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8305 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8306 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8307 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8308 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8309 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8310 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8311 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8312 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8313 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8314 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8315 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8316 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8317 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8318 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8319 := (leafCircuit_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8320 := (leafCircuit_copies260 a h).1
  have c8321 := (leafCircuit_copies260 a h).2.1
  have c8322 := (leafCircuit_copies260 a h).2.2.1
  have c8323 := (leafCircuit_copies260 a h).2.2.2.1
  have c8324 := (leafCircuit_copies260 a h).2.2.2.2.1
  have c8325 := (leafCircuit_copies260 a h).2.2.2.2.2.1
  have c8326 := (leafCircuit_copies260 a h).2.2.2.2.2.2.1
  have c8327 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.1
  have c8328 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.1
  have c8329 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.1
  have c8330 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8331 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8332 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8333 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8334 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8335 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_203_6 := arithEq_of_rows h (row := 203) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_6
  simp only [← c8162, ← c8163, k3, ← c8164] at e_203_6
  have e_203_7 := arithEq_of_rows h (row := 203) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_7
  simp only [← c8165, ← c8166, k3, ← c8167] at e_203_7
  have e_203_8 := arithEq_of_rows h (row := 203) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_8
  simp only [← c8168, ← c8169, k3, ← c8170] at e_203_8
  have e_203_9 := arithEq_of_rows h (row := 203) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_9
  simp only [← c8171, ← c8172, k3, ← c8173] at e_203_9
  have e_203_10 := arithEq_of_rows h (row := 203) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_10
  simp only [← c8174, ← c8175, k3, ← c8176] at e_203_10
  have e_203_11 := arithEq_of_rows h (row := 203) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_11
  simp only [← c8177, ← c8178, k3, ← c8179] at e_203_11
  have e_203_12 := arithEq_of_rows h (row := 203) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_12
  simp only [← c8180, ← c8181, k3, ← c8182] at e_203_12
  have e_203_13 := arithEq_of_rows h (row := 203) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_13
  simp only [← c8183, ← c8184, k3, ← c8185] at e_203_13
  have e_203_14 := arithEq_of_rows h (row := 203) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_14
  simp only [← c8198, ← c8199, k3, ← c8200] at e_203_14
  have e_215_0 := arithEq_of_rows h (row := 215) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_0
  simp only [← c8201, ← c8202, k3, ← c8203] at e_215_0
  have e_215_1 := arithEq_of_rows h (row := 215) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_1
  simp only [← c8204, ← c8205, k3, ← c8206] at e_215_1
  have e_215_2 := arithEq_of_rows h (row := 215) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_2
  simp only [← c8207, ← c8208, k3, ← c8209] at e_215_2
  have e_215_3 := arithEq_of_rows h (row := 215) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_3
  simp only [← c8210, ← c8211, k3, ← c8212] at e_215_3
  have e_215_4 := arithEq_of_rows h (row := 215) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_4
  simp only [← c8213, ← c8214, k3, ← c8215] at e_215_4
  have e_215_5 := arithEq_of_rows h (row := 215) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_5
  simp only [← c8216, ← c8217, k3, ← c8218] at e_215_5
  have e_215_6 := arithEq_of_rows h (row := 215) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_6
  simp only [← c8219, ← c8220, k3, ← c8221] at e_215_6
  have e_215_7 := arithEq_of_rows h (row := 215) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_7
  simp only [← c8234, ← c8235, k3, ← c8236] at e_215_7
  have e_215_8 := arithEq_of_rows h (row := 215) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_8
  simp only [← c8237, ← c8238, k3, ← c8239] at e_215_8
  have e_215_9 := arithEq_of_rows h (row := 215) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_9
  simp only [← c8240, ← c8241, k3, ← c8242] at e_215_9
  have e_215_10 := arithEq_of_rows h (row := 215) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_10
  simp only [← c8243, ← c8244, k3, ← c8245] at e_215_10
  have e_215_11 := arithEq_of_rows h (row := 215) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_11
  simp only [← c8246, ← c8247, k3, ← c8248] at e_215_11
  have e_215_12 := arithEq_of_rows h (row := 215) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_12
  simp only [← c8249, ← c8250, k3, ← c8251] at e_215_12
  have e_215_13 := arithEq_of_rows h (row := 215) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_13
  simp only [← c8252, ← c8253, k3, ← c8254] at e_215_13
  have e_215_14 := arithEq_of_rows h (row := 215) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_14
  simp only [← c8255, ← c8256, k3, ← c8257] at e_215_14
  have e_218_0 := arithEq_of_rows h (row := 218) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_0
  simp only [← c8270, ← c8271, k3, ← c8272] at e_218_0
  have e_218_1 := arithEq_of_rows h (row := 218) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_1
  simp only [← c8273, ← c8274, k3, ← c8275] at e_218_1
  have e_218_2 := arithEq_of_rows h (row := 218) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_2
  simp only [← c8276, ← c8277, k3, ← c8278] at e_218_2
  have e_218_3 := arithEq_of_rows h (row := 218) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_3
  simp only [← c8279, ← c8280, k3, ← c8281] at e_218_3
  have e_218_4 := arithEq_of_rows h (row := 218) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_4
  simp only [← c8282, ← c8283, k3, ← c8284] at e_218_4
  have e_218_5 := arithEq_of_rows h (row := 218) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_5
  simp only [← c8285, ← c8286, k3, ← c8287] at e_218_5
  have e_218_6 := arithEq_of_rows h (row := 218) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_6
  simp only [← c8288, ← c8289, k3, ← c8290] at e_218_6
  have e_218_7 := arithEq_of_rows h (row := 218) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_7
  simp only [← c8291, ← c8292, k3, ← c8293] at e_218_7
  have e_218_8 := arithEq_of_rows h (row := 218) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_8
  simp only [← c8306, ← c8307, k3, ← c8308] at e_218_8
  have e_218_9 := arithEq_of_rows h (row := 218) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_9
  simp only [← c8309, ← c8310, k3, ← c8311] at e_218_9
  have e_218_10 := arithEq_of_rows h (row := 218) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_10
  simp only [← c8312, ← c8313, k3, ← c8314] at e_218_10
  have e_218_11 := arithEq_of_rows h (row := 218) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_11
  simp only [← c8315, ← c8316, k3, ← c8317] at e_218_11
  have e_218_12 := arithEq_of_rows h (row := 218) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_12
  simp only [← c8318, ← c8319, k3, ← c8320] at e_218_12
  have e_218_13 := arithEq_of_rows h (row := 218) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_13
  simp only [← c8321, ← c8322, k3, ← c8323, k3] at e_218_13
  have hin0 : poseidon2In a 213 = addBlock (fun _ => 0) [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252)] :=
    poseidon2In_first c8150.symm c8151.symm c8152.symm c8153.symm c8154.symm c8155.symm c8156.symm c8157.symm (c8158.symm.trans k0) (c8159.symm.trans k0) (c8160.symm.trans k0) (c8161.symm.trans k0)
  have hout0 := poseidon2Row_absorb perm hp (row := 213) rfl rfl hin0
  have hin1 : poseidon2In a 214 = addBlock (poseidon2Out a 213) [a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260)] :=
    poseidon2In_chain (by linear_combination -c8186 + e_203_6) (by linear_combination -c8187 + e_203_7) (by linear_combination -c8188 + e_203_8) (by linear_combination -c8189 + e_203_9) (by linear_combination -c8190 + e_203_10) (by linear_combination -c8191 + e_203_11) (by linear_combination -c8192 + e_203_12) (by linear_combination -c8193 + e_203_13) c8194.symm c8195.symm c8196.symm c8197.symm
  have hout1 := poseidon2Row_absorb perm hp (row := 214) rfl rfl hin1
  have hin2 : poseidon2In a 216 = addBlock (poseidon2Out a 214) [a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268)] :=
    poseidon2In_chain (by linear_combination -c8222 + e_203_14) (by linear_combination -c8223 + e_215_0) (by linear_combination -c8224 + e_215_1) (by linear_combination -c8225 + e_215_2) (by linear_combination -c8226 + e_215_3) (by linear_combination -c8227 + e_215_4) (by linear_combination -c8228 + e_215_5) (by linear_combination -c8229 + e_215_6) c8230.symm c8231.symm c8232.symm c8233.symm
  have hout2 := poseidon2Row_absorb perm hp (row := 216) rfl rfl hin2
  have hin3 : poseidon2In a 217 = addBlock (poseidon2Out a 216) [a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276)] :=
    poseidon2In_chain (by linear_combination -c8258 + e_215_7) (by linear_combination -c8259 + e_215_8) (by linear_combination -c8260 + e_215_9) (by linear_combination -c8261 + e_215_10) (by linear_combination -c8262 + e_215_11) (by linear_combination -c8263 + e_215_12) (by linear_combination -c8264 + e_215_13) (by linear_combination -c8265 + e_215_14) c8266.symm c8267.symm c8268.symm c8269.symm
  have hout3 := poseidon2Row_absorb perm hp (row := 217) rfl rfl hin3
  have hin4 : poseidon2In a 219 = addBlock (poseidon2Out a 217) [a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284)] :=
    poseidon2In_chain (by linear_combination -c8294 + e_218_0) (by linear_combination -c8295 + e_218_1) (by linear_combination -c8296 + e_218_2) (by linear_combination -c8297 + e_218_3) (by linear_combination -c8298 + e_218_4) (by linear_combination -c8299 + e_218_5) (by linear_combination -c8300 + e_218_6) (by linear_combination -c8301 + e_218_7) c8302.symm c8303.symm c8304.symm c8305.symm
  have hout4 := poseidon2Row_absorb perm hp (row := 219) rfl rfl hin4
  have hin5 : poseidon2In a 220 = addBlock (poseidon2Out a 219) [a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289), 1, 0, 0] :=
    poseidon2In_chain (by linear_combination -c8324 + e_218_8) (by linear_combination -c8325 + e_218_9) (by linear_combination -c8326 + e_218_10) (by linear_combination -c8327 + e_218_11) (by linear_combination -c8328 + e_218_12) (by linear_combination -c8329 + e_218_13) (c8330.symm.trans (add_zero _).symm) (c8331.symm.trans (add_zero _).symm) c8332.symm c8333.symm c8334.symm c8335.symm
  have hout5 := poseidon2Row_absorb perm hp (row := 220) rfl rfl hin5
  have hpad : pad10 [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] =
      [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252)] ++ ([a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260)] ++ ([a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268)] ++ ([a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276)] ++ ([a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284)] ++ ([a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289), 1, 0, 0] ++ []))))) := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_block8, absorbMsg_nil, ← hout0, ← hout1, ← hout2, ← hout3, ← hout4, ← hout5]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leafCircuit_f1537 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 35) = a (.virt 241) - a (.wire 220 12) := by
  have c8336 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8337 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8338 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_8 := arithEq_of_rows h (row := 209) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_8
  simp only [← c8336, ← c8337, k3, ← c8338] at e_209_8
  have hr := e_209_8
  linear_combination hr

theorem leafCircuit_f1538 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 3) = a (.wire 209 35) * a (.wire 209 15) := by
  have c8339 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8340 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8341 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_0 := arithEq_of_rows h (row := 221) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_0
  simp only [← c8339, ← c8340, ← c8341] at e_221_0
  have hr := e_221_0
  linear_combination hr

theorem leafCircuit_f1539 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 3) = a (.virt 14) := by
  have c8342 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8342

theorem leafCircuit_f1540 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 39) = a (.virt 242) - a (.wire 220 13) := by
  have c8343 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8344 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8345 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_9 := arithEq_of_rows h (row := 209) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_9
  simp only [← c8343, ← c8344, k3, ← c8345] at e_209_9
  have hr := e_209_9
  linear_combination hr

theorem leafCircuit_f1541 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 7) = a (.wire 209 39) * a (.wire 209 15) := by
  have c8346 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8347 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8348 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_1 := arithEq_of_rows h (row := 221) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_1
  simp only [← c8346, ← c8347, ← c8348] at e_221_1
  have hr := e_221_1
  linear_combination hr

theorem leafCircuit_f1542 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 7) = a (.virt 14) := by
  have c8349 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8349

theorem leafCircuit_f1543 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 43) = a (.virt 243) - a (.wire 220 14) := by
  have c8350 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8351 := (leafCircuit_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8352 := (leafCircuit_copies261 a h).1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_10 := arithEq_of_rows h (row := 209) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_10
  simp only [← c8350, ← c8351, k3, ← c8352] at e_209_10
  have hr := e_209_10
  linear_combination hr

theorem leafCircuit_f1544 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 11) = a (.wire 209 43) * a (.wire 209 15) := by
  have c8353 := (leafCircuit_copies261 a h).2.1
  have c8354 := (leafCircuit_copies261 a h).2.2.1
  have c8355 := (leafCircuit_copies261 a h).2.2.2.1
  have e_221_2 := arithEq_of_rows h (row := 221) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_2
  simp only [← c8353, ← c8354, ← c8355] at e_221_2
  have hr := e_221_2
  linear_combination hr

theorem leafCircuit_f1545 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 11) = a (.virt 14) := by
  have c8356 := (leafCircuit_copies261 a h).2.2.2.2.1
  exact c8356

theorem leafCircuit_f1546 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 47) = a (.virt 244) - a (.wire 220 15) := by
  have c8357 := (leafCircuit_copies261 a h).2.2.2.2.2.1
  have c8358 := (leafCircuit_copies261 a h).2.2.2.2.2.2.1
  have c8359 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_11 := arithEq_of_rows h (row := 209) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_11
  simp only [← c8357, ← c8358, k3, ← c8359] at e_209_11
  have hr := e_209_11
  linear_combination hr

theorem leafCircuit_f1547 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 15) = a (.wire 209 47) * a (.wire 209 15) := by
  have c8360 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.1
  have c8361 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.1
  have c8362 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_221_3 := arithEq_of_rows h (row := 221) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_3
  simp only [← c8360, ← c8361, ← c8362] at e_221_3
  have hr := e_221_3
  linear_combination hr

theorem leafCircuit_f1548 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 15) = a (.virt 14) := by
  have c8363 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c8363

theorem leafCircuit_f1549 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 51) = a (.virt 258) - a (.virt 8) := by
  have c8364 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8365 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8366 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_12 := arithEq_of_rows h (row := 209) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_12
  simp only [← c8364, ← c8365, k3, ← c8366] at e_209_12
  have hr := e_209_12
  linear_combination hr

theorem leafCircuit_f1550 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 19) = a (.wire 209 51) * a (.wire 209 15) := by
  have c8367 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8368 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8369 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_4 := arithEq_of_rows h (row := 221) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_4
  simp only [← c8367, ← c8368, ← c8369] at e_221_4
  have hr := e_221_4
  linear_combination hr

theorem leafCircuit_f1551 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 19) = a (.virt 14) := by
  have c8370 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8370

theorem leafCircuit_f1552 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 55) = a (.virt 259) - a (.virt 9) := by
  have c8371 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8372 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8373 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_13 := arithEq_of_rows h (row := 209) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_13
  simp only [← c8371, ← c8372, k3, ← c8373] at e_209_13
  have hr := e_209_13
  linear_combination hr

theorem leafCircuit_f1553 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 23) = a (.wire 209 55) * a (.wire 209 15) := by
  have c8374 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8375 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8376 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_5 := arithEq_of_rows h (row := 221) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_5
  simp only [← c8374, ← c8375, ← c8376] at e_221_5
  have hr := e_221_5
  linear_combination hr

theorem leafCircuit_f1554 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 23) = a (.virt 14) := by
  have c8377 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c8377

theorem leafCircuit_f1555 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 209 59) = a (.virt 260) - a (.virt 10) := by
  have c8378 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8379 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8380 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_209_14 := arithEq_of_rows h (row := 209) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_14
  simp only [← c8378, ← c8379, k3, ← c8380] at e_209_14
  have hr := e_209_14
  linear_combination hr

theorem leafCircuit_f1556 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 27) = a (.wire 209 59) * a (.wire 209 15) := by
  have c8381 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8382 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8383 := (leafCircuit_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_221_6 := arithEq_of_rows h (row := 221) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_6
  simp only [← c8381, ← c8382, ← c8383] at e_221_6
  have hr := e_221_6
  linear_combination hr

theorem leafCircuit_f1557 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 27) = a (.virt 14) := by
  have c8384 := (leafCircuit_copies262 a h).1
  exact c8384

theorem leafCircuit_f1558 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 222 3) = a (.virt 261) - a (.virt 11) := by
  have c8385 := (leafCircuit_copies262 a h).2.1
  have c8386 := (leafCircuit_copies262 a h).2.2.1
  have c8387 := (leafCircuit_copies262 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4, k5, k6, k7, k8⟩ := leafCircuit_consts a h
  have e_222_0 := arithEq_of_rows h (row := 222) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_0
  simp only [← c8385, ← c8386, k3, ← c8387] at e_222_0
  have hr := e_222_0
  linear_combination hr

theorem leafCircuit_f1559 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 31) = a (.wire 222 3) * a (.wire 209 15) := by
  have c8388 := (leafCircuit_copies262 a h).2.2.2.2.1
  have c8389 := (leafCircuit_copies262 a h).2.2.2.2.2.1
  have c8390 := (leafCircuit_copies262 a h).2.2.2.2.2.2.1
  have e_221_7 := arithEq_of_rows h (row := 221) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_7
  simp only [← c8388, ← c8389, ← c8390] at e_221_7
  have hr := e_221_7
  linear_combination hr

theorem leafCircuit_f1560 (a : Assignment p) (h : Satisfies (leafCircuit p) a) :
    a (.wire 221 31) = a (.virt 14) := by
  have c8391 := (leafCircuit_copies262 a h).2.2.2.2.2.2.2
  exact c8391

end Plonky2Spec.Generated
