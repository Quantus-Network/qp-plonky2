/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PublicBatchWrapper86`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.Generated.PublicBatchWrapper86.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem publicBatchWrapper86_f9800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1020 71) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8566)) := by
  have c63153 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63154 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63155 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63156 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63157 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63158 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1020_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_16
  simp only [← c63153, ← c63154, ← c63155] at e_1020_16
  have e_1020_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_17
  simp only [← c63156, ← c63157, k2, ← c63158] at e_1020_17
  have hr := e_1020_17
  simp only [e_1020_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1020 79) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8567)) := by
  have c63159 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63160 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63161 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63162 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63163 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63164 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1020_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_18
  simp only [← c63159, ← c63160, ← c63161] at e_1020_18
  have e_1020_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_19
  simp only [← c63162, ← c63163, k2, ← c63164] at e_1020_19
  have hr := e_1020_19
  simp only [e_1020_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 7) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8568)) := by
  have c63165 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63166 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63167 := (publicBatchWrapper86_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63168 := (publicBatchWrapper86_copies1974 a h).1
  have c63169 := (publicBatchWrapper86_copies1974 a h).2.1
  have c63170 := (publicBatchWrapper86_copies1974 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_0
  simp only [← c63165, ← c63166, ← c63167] at e_1021_0
  have e_1021_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_1
  simp only [← c63168, ← c63169, k2, ← c63170] at e_1021_1
  have hr := e_1021_1
  simp only [e_1021_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 15) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8569)) := by
  have c63171 := (publicBatchWrapper86_copies1974 a h).2.2.2.1
  have c63172 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.1
  have c63173 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.1
  have c63174 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.1
  have c63175 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.1
  have c63176 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_2
  simp only [← c63171, ← c63172, ← c63173] at e_1021_2
  have e_1021_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_3
  simp only [← c63174, ← c63175, k2, ← c63176] at e_1021_3
  have hr := e_1021_3
  simp only [e_1021_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8610)) := by
  have c63177 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.1
  have c63178 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63179 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63180 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63181 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63182 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_4
  simp only [← c63177, ← c63178, ← c63179] at e_1021_4
  have e_1021_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_5
  simp only [← c63180, ← c63181, k2, ← c63182] at e_1021_5
  have hr := e_1021_5
  simp only [e_1021_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8611)) := by
  have c63183 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63184 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63185 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63186 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63187 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63188 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_6
  simp only [← c63183, ← c63184, ← c63185] at e_1021_6
  have e_1021_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_7
  simp only [← c63186, ← c63187, k2, ← c63188] at e_1021_7
  have hr := e_1021_7
  simp only [e_1021_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8612)) := by
  have c63189 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63190 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63191 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63192 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63193 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63194 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_8
  simp only [← c63189, ← c63190, ← c63191] at e_1021_8
  have e_1021_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_9
  simp only [← c63192, ← c63193, k2, ← c63194] at e_1021_9
  have hr := e_1021_9
  simp only [e_1021_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8613)) := by
  have c63195 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63196 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63197 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63198 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63199 := (publicBatchWrapper86_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63200 := (publicBatchWrapper86_copies1975 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_10
  simp only [← c63195, ← c63196, ← c63197] at e_1021_10
  have e_1021_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_11
  simp only [← c63198, ← c63199, k2, ← c63200] at e_1021_11
  have hr := e_1021_11
  simp only [e_1021_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8614)) := by
  have c63201 := (publicBatchWrapper86_copies1975 a h).2.1
  have c63202 := (publicBatchWrapper86_copies1975 a h).2.2.1
  have c63203 := (publicBatchWrapper86_copies1975 a h).2.2.2.1
  have c63204 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.1
  have c63205 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.1
  have c63206 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_12
  simp only [← c63201, ← c63202, ← c63203] at e_1021_12
  have e_1021_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_13
  simp only [← c63204, ← c63205, k2, ← c63206] at e_1021_13
  have hr := e_1021_13
  simp only [e_1021_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8615)) := by
  have c63207 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.1
  have c63208 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.1
  have c63209 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.1
  have c63210 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63211 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63212 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_14
  simp only [← c63207, ← c63208, ← c63209] at e_1021_14
  have e_1021_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_15
  simp only [← c63210, ← c63211, k2, ← c63212] at e_1021_15
  have hr := e_1021_15
  simp only [e_1021_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8616)) := by
  have c63213 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63214 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63215 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63216 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63217 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63218 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_16
  simp only [← c63213, ← c63214, ← c63215] at e_1021_16
  have e_1021_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_17
  simp only [← c63216, ← c63217, k2, ← c63218] at e_1021_17
  have hr := e_1021_17
  simp only [e_1021_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1021 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8617)) := by
  have c63219 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63220 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63221 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63222 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63223 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63224 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1021_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_18
  simp only [← c63219, ← c63220, ← c63221] at e_1021_18
  have e_1021_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_19
  simp only [← c63222, ← c63223, k2, ← c63224] at e_1021_19
  have hr := e_1021_19
  simp only [e_1021_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8618)) := by
  have c63225 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63226 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63227 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63228 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63229 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63230 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_0
  simp only [← c63225, ← c63226, ← c63227] at e_1022_0
  have e_1022_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_1
  simp only [← c63228, ← c63229, k2, ← c63230] at e_1022_1
  have hr := e_1022_1
  simp only [e_1022_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8619)) := by
  have c63231 := (publicBatchWrapper86_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63232 := (publicBatchWrapper86_copies1976 a h).1
  have c63233 := (publicBatchWrapper86_copies1976 a h).2.1
  have c63234 := (publicBatchWrapper86_copies1976 a h).2.2.1
  have c63235 := (publicBatchWrapper86_copies1976 a h).2.2.2.1
  have c63236 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_2
  simp only [← c63231, ← c63232, ← c63233] at e_1022_2
  have e_1022_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_3
  simp only [← c63234, ← c63235, k2, ← c63236] at e_1022_3
  have hr := e_1022_3
  simp only [e_1022_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8620)) := by
  have c63237 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.1
  have c63238 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.1
  have c63239 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.1
  have c63240 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.1
  have c63241 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.1
  have c63242 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_4
  simp only [← c63237, ← c63238, ← c63239] at e_1022_4
  have e_1022_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_5
  simp only [← c63240, ← c63241, k2, ← c63242] at e_1022_5
  have hr := e_1022_5
  simp only [e_1022_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8621)) := by
  have c63243 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63244 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63245 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63246 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63247 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63248 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_6
  simp only [← c63243, ← c63244, ← c63245] at e_1022_6
  have e_1022_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_7
  simp only [← c63246, ← c63247, k2, ← c63248] at e_1022_7
  have hr := e_1022_7
  simp only [e_1022_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8622)) := by
  have c63249 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63250 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63251 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63252 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63253 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63254 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_8
  simp only [← c63249, ← c63250, ← c63251] at e_1022_8
  have e_1022_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_9
  simp only [← c63252, ← c63253, k2, ← c63254] at e_1022_9
  have hr := e_1022_9
  simp only [e_1022_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8623)) := by
  have c63255 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63256 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63257 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63258 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63259 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63260 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_10
  simp only [← c63255, ← c63256, ← c63257] at e_1022_10
  have e_1022_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_11
  simp only [← c63258, ← c63259, k2, ← c63260] at e_1022_11
  have hr := e_1022_11
  simp only [e_1022_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8624)) := by
  have c63261 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63262 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63263 := (publicBatchWrapper86_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63264 := (publicBatchWrapper86_copies1977 a h).1
  have c63265 := (publicBatchWrapper86_copies1977 a h).2.1
  have c63266 := (publicBatchWrapper86_copies1977 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_12
  simp only [← c63261, ← c63262, ← c63263] at e_1022_12
  have e_1022_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_13
  simp only [← c63264, ← c63265, k2, ← c63266] at e_1022_13
  have hr := e_1022_13
  simp only [e_1022_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8625)) := by
  have c63267 := (publicBatchWrapper86_copies1977 a h).2.2.2.1
  have c63268 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.1
  have c63269 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.1
  have c63270 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.1
  have c63271 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.1
  have c63272 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_14
  simp only [← c63267, ← c63268, ← c63269] at e_1022_14
  have e_1022_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_15
  simp only [← c63270, ← c63271, k2, ← c63272] at e_1022_15
  have hr := e_1022_15
  simp only [e_1022_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8626)) := by
  have c63273 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.1
  have c63274 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63275 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63276 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63277 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63278 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_16
  simp only [← c63273, ← c63274, ← c63275] at e_1022_16
  have e_1022_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_17
  simp only [← c63276, ← c63277, k2, ← c63278] at e_1022_17
  have hr := e_1022_17
  simp only [e_1022_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1022 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8627)) := by
  have c63279 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63280 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63281 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63282 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63283 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63284 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1022_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_18
  simp only [← c63279, ← c63280, ← c63281] at e_1022_18
  have e_1022_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_19
  simp only [← c63282, ← c63283, k2, ← c63284] at e_1022_19
  have hr := e_1022_19
  simp only [e_1022_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8628)) := by
  have c63285 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63286 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63287 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63288 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63289 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63290 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_0
  simp only [← c63285, ← c63286, ← c63287] at e_1023_0
  have e_1023_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_1
  simp only [← c63288, ← c63289, k2, ← c63290] at e_1023_1
  have hr := e_1023_1
  simp only [e_1023_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8629)) := by
  have c63291 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63292 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63293 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63294 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63295 := (publicBatchWrapper86_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63296 := (publicBatchWrapper86_copies1978 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_2
  simp only [← c63291, ← c63292, ← c63293] at e_1023_2
  have e_1023_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_3
  simp only [← c63294, ← c63295, k2, ← c63296] at e_1023_3
  have hr := e_1023_3
  simp only [e_1023_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8630)) := by
  have c63297 := (publicBatchWrapper86_copies1978 a h).2.1
  have c63298 := (publicBatchWrapper86_copies1978 a h).2.2.1
  have c63299 := (publicBatchWrapper86_copies1978 a h).2.2.2.1
  have c63300 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.1
  have c63301 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.1
  have c63302 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_4
  simp only [← c63297, ← c63298, ← c63299] at e_1023_4
  have e_1023_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_5
  simp only [← c63300, ← c63301, k2, ← c63302] at e_1023_5
  have hr := e_1023_5
  simp only [e_1023_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8631)) := by
  have c63303 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.1
  have c63304 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.1
  have c63305 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.1
  have c63306 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63307 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63308 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_6
  simp only [← c63303, ← c63304, ← c63305] at e_1023_6
  have e_1023_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_7
  simp only [← c63306, ← c63307, k2, ← c63308] at e_1023_7
  have hr := e_1023_7
  simp only [e_1023_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8632)) := by
  have c63309 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63310 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63311 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63312 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63313 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63314 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_8
  simp only [← c63309, ← c63310, ← c63311] at e_1023_8
  have e_1023_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_9
  simp only [← c63312, ← c63313, k2, ← c63314] at e_1023_9
  have hr := e_1023_9
  simp only [e_1023_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8633)) := by
  have c63315 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63316 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63317 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63318 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63319 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63320 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_10
  simp only [← c63315, ← c63316, ← c63317] at e_1023_10
  have e_1023_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_11
  simp only [← c63318, ← c63319, k2, ← c63320] at e_1023_11
  have hr := e_1023_11
  simp only [e_1023_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8634)) := by
  have c63321 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63322 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63323 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63324 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63325 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63326 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_12
  simp only [← c63321, ← c63322, ← c63323] at e_1023_12
  have e_1023_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_13
  simp only [← c63324, ← c63325, k2, ← c63326] at e_1023_13
  have hr := e_1023_13
  simp only [e_1023_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8635)) := by
  have c63327 := (publicBatchWrapper86_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63328 := (publicBatchWrapper86_copies1979 a h).1
  have c63329 := (publicBatchWrapper86_copies1979 a h).2.1
  have c63330 := (publicBatchWrapper86_copies1979 a h).2.2.1
  have c63331 := (publicBatchWrapper86_copies1979 a h).2.2.2.1
  have c63332 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_14
  simp only [← c63327, ← c63328, ← c63329] at e_1023_14
  have e_1023_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_15
  simp only [← c63330, ← c63331, k2, ← c63332] at e_1023_15
  have hr := e_1023_15
  simp only [e_1023_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8636)) := by
  have c63333 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.1
  have c63334 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.1
  have c63335 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.1
  have c63336 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.1
  have c63337 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.1
  have c63338 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_16
  simp only [← c63333, ← c63334, ← c63335] at e_1023_16
  have e_1023_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_17
  simp only [← c63336, ← c63337, k2, ← c63338] at e_1023_17
  have hr := e_1023_17
  simp only [e_1023_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1023 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8637)) := by
  have c63339 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63340 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63341 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63342 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63343 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63344 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1023_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_18
  simp only [← c63339, ← c63340, ← c63341] at e_1023_18
  have e_1023_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_19
  simp only [← c63342, ← c63343, k2, ← c63344] at e_1023_19
  have hr := e_1023_19
  simp only [e_1023_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8638)) := by
  have c63345 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63346 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63347 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63348 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63349 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63350 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_0
  simp only [← c63345, ← c63346, ← c63347] at e_1024_0
  have e_1024_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_1
  simp only [← c63348, ← c63349, k2, ← c63350] at e_1024_1
  have hr := e_1024_1
  simp only [e_1024_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8639)) := by
  have c63351 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63352 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63353 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63354 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63355 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63356 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_2
  simp only [← c63351, ← c63352, ← c63353] at e_1024_2
  have e_1024_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_3
  simp only [← c63354, ← c63355, k2, ← c63356] at e_1024_3
  have hr := e_1024_3
  simp only [e_1024_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8640)) := by
  have c63357 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63358 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63359 := (publicBatchWrapper86_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63360 := (publicBatchWrapper86_copies1980 a h).1
  have c63361 := (publicBatchWrapper86_copies1980 a h).2.1
  have c63362 := (publicBatchWrapper86_copies1980 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_4
  simp only [← c63357, ← c63358, ← c63359] at e_1024_4
  have e_1024_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_5
  simp only [← c63360, ← c63361, k2, ← c63362] at e_1024_5
  have hr := e_1024_5
  simp only [e_1024_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8641)) := by
  have c63363 := (publicBatchWrapper86_copies1980 a h).2.2.2.1
  have c63364 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.1
  have c63365 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.1
  have c63366 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.1
  have c63367 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.1
  have c63368 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_6
  simp only [← c63363, ← c63364, ← c63365] at e_1024_6
  have e_1024_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_7
  simp only [← c63366, ← c63367, k2, ← c63368] at e_1024_7
  have hr := e_1024_7
  simp only [e_1024_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8642)) := by
  have c63369 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.1
  have c63370 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63371 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63372 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63373 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63374 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_8
  simp only [← c63369, ← c63370, ← c63371] at e_1024_8
  have e_1024_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_9
  simp only [← c63372, ← c63373, k2, ← c63374] at e_1024_9
  have hr := e_1024_9
  simp only [e_1024_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8643)) := by
  have c63375 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63376 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63377 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63378 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63379 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63380 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_10
  simp only [← c63375, ← c63376, ← c63377] at e_1024_10
  have e_1024_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_11
  simp only [← c63378, ← c63379, k2, ← c63380] at e_1024_11
  have hr := e_1024_11
  simp only [e_1024_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8644)) := by
  have c63381 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63382 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63383 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63384 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63385 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63386 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_12
  simp only [← c63381, ← c63382, ← c63383] at e_1024_12
  have e_1024_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_13
  simp only [← c63384, ← c63385, k2, ← c63386] at e_1024_13
  have hr := e_1024_13
  simp only [e_1024_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8645)) := by
  have c63387 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63388 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63389 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63390 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63391 := (publicBatchWrapper86_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63392 := (publicBatchWrapper86_copies1981 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_14
  simp only [← c63387, ← c63388, ← c63389] at e_1024_14
  have e_1024_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_15
  simp only [← c63390, ← c63391, k2, ← c63392] at e_1024_15
  have hr := e_1024_15
  simp only [e_1024_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8646)) := by
  have c63393 := (publicBatchWrapper86_copies1981 a h).2.1
  have c63394 := (publicBatchWrapper86_copies1981 a h).2.2.1
  have c63395 := (publicBatchWrapper86_copies1981 a h).2.2.2.1
  have c63396 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.1
  have c63397 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.1
  have c63398 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_16
  simp only [← c63393, ← c63394, ← c63395] at e_1024_16
  have e_1024_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_17
  simp only [← c63396, ← c63397, k2, ← c63398] at e_1024_17
  have hr := e_1024_17
  simp only [e_1024_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1024 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8647)) := by
  have c63399 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.1
  have c63400 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.1
  have c63401 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.1
  have c63402 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63403 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63404 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1024_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_18
  simp only [← c63399, ← c63400, ← c63401] at e_1024_18
  have e_1024_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_19
  simp only [← c63402, ← c63403, k2, ← c63404] at e_1024_19
  have hr := e_1024_19
  simp only [e_1024_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8648)) := by
  have c63405 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63406 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63407 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63408 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63409 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63410 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_0
  simp only [← c63405, ← c63406, ← c63407] at e_1025_0
  have e_1025_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_1
  simp only [← c63408, ← c63409, k2, ← c63410] at e_1025_1
  have hr := e_1025_1
  simp only [e_1025_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8649)) := by
  have c63411 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63412 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63413 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63414 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63415 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63416 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_2
  simp only [← c63411, ← c63412, ← c63413] at e_1025_2
  have e_1025_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_3
  simp only [← c63414, ← c63415, k2, ← c63416] at e_1025_3
  have hr := e_1025_3
  simp only [e_1025_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8650)) := by
  have c63417 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63418 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63419 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63420 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63421 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63422 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_4
  simp only [← c63417, ← c63418, ← c63419] at e_1025_4
  have e_1025_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_5
  simp only [← c63420, ← c63421, k2, ← c63422] at e_1025_5
  have hr := e_1025_5
  simp only [e_1025_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8651)) := by
  have c63423 := (publicBatchWrapper86_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63424 := (publicBatchWrapper86_copies1982 a h).1
  have c63425 := (publicBatchWrapper86_copies1982 a h).2.1
  have c63426 := (publicBatchWrapper86_copies1982 a h).2.2.1
  have c63427 := (publicBatchWrapper86_copies1982 a h).2.2.2.1
  have c63428 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_6
  simp only [← c63423, ← c63424, ← c63425] at e_1025_6
  have e_1025_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_7
  simp only [← c63426, ← c63427, k2, ← c63428] at e_1025_7
  have hr := e_1025_7
  simp only [e_1025_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8652)) := by
  have c63429 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.1
  have c63430 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.1
  have c63431 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.1
  have c63432 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.1
  have c63433 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.1
  have c63434 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_8
  simp only [← c63429, ← c63430, ← c63431] at e_1025_8
  have e_1025_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_9
  simp only [← c63432, ← c63433, k2, ← c63434] at e_1025_9
  have hr := e_1025_9
  simp only [e_1025_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8653)) := by
  have c63435 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63436 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63437 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63438 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63439 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63440 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_10
  simp only [← c63435, ← c63436, ← c63437] at e_1025_10
  have e_1025_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_11
  simp only [← c63438, ← c63439, k2, ← c63440] at e_1025_11
  have hr := e_1025_11
  simp only [e_1025_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8654)) := by
  have c63441 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63442 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63443 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63444 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63445 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63446 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_12
  simp only [← c63441, ← c63442, ← c63443] at e_1025_12
  have e_1025_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_13
  simp only [← c63444, ← c63445, k2, ← c63446] at e_1025_13
  have hr := e_1025_13
  simp only [e_1025_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8655)) := by
  have c63447 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63448 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63449 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63450 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63451 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63452 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_14
  simp only [← c63447, ← c63448, ← c63449] at e_1025_14
  have e_1025_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_15
  simp only [← c63450, ← c63451, k2, ← c63452] at e_1025_15
  have hr := e_1025_15
  simp only [e_1025_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8656)) := by
  have c63453 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63454 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63455 := (publicBatchWrapper86_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63456 := (publicBatchWrapper86_copies1983 a h).1
  have c63457 := (publicBatchWrapper86_copies1983 a h).2.1
  have c63458 := (publicBatchWrapper86_copies1983 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_16
  simp only [← c63453, ← c63454, ← c63455] at e_1025_16
  have e_1025_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_17
  simp only [← c63456, ← c63457, k2, ← c63458] at e_1025_17
  have hr := e_1025_17
  simp only [e_1025_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1025 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8657)) := by
  have c63459 := (publicBatchWrapper86_copies1983 a h).2.2.2.1
  have c63460 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.1
  have c63461 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.1
  have c63462 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.1
  have c63463 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.1
  have c63464 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1025_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_18
  simp only [← c63459, ← c63460, ← c63461] at e_1025_18
  have e_1025_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_19
  simp only [← c63462, ← c63463, k2, ← c63464] at e_1025_19
  have hr := e_1025_19
  simp only [e_1025_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8658)) := by
  have c63465 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.1
  have c63466 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63467 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63468 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63469 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63470 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_0
  simp only [← c63465, ← c63466, ← c63467] at e_1026_0
  have e_1026_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_1
  simp only [← c63468, ← c63469, k2, ← c63470] at e_1026_1
  have hr := e_1026_1
  simp only [e_1026_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8659)) := by
  have c63471 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63472 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63473 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63474 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63475 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63476 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_2
  simp only [← c63471, ← c63472, ← c63473] at e_1026_2
  have e_1026_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_3
  simp only [← c63474, ← c63475, k2, ← c63476] at e_1026_3
  have hr := e_1026_3
  simp only [e_1026_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8660)) := by
  have c63477 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63478 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63479 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63480 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63481 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63482 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_4
  simp only [← c63477, ← c63478, ← c63479] at e_1026_4
  have e_1026_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_5
  simp only [← c63480, ← c63481, k2, ← c63482] at e_1026_5
  have hr := e_1026_5
  simp only [e_1026_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8661)) := by
  have c63483 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63484 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63485 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63486 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63487 := (publicBatchWrapper86_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63488 := (publicBatchWrapper86_copies1984 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_6
  simp only [← c63483, ← c63484, ← c63485] at e_1026_6
  have e_1026_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_7
  simp only [← c63486, ← c63487, k2, ← c63488] at e_1026_7
  have hr := e_1026_7
  simp only [e_1026_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8662)) := by
  have c63489 := (publicBatchWrapper86_copies1984 a h).2.1
  have c63490 := (publicBatchWrapper86_copies1984 a h).2.2.1
  have c63491 := (publicBatchWrapper86_copies1984 a h).2.2.2.1
  have c63492 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.1
  have c63493 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.1
  have c63494 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_8
  simp only [← c63489, ← c63490, ← c63491] at e_1026_8
  have e_1026_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_9
  simp only [← c63492, ← c63493, k2, ← c63494] at e_1026_9
  have hr := e_1026_9
  simp only [e_1026_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8663)) := by
  have c63495 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.1
  have c63496 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.1
  have c63497 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.1
  have c63498 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63499 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63500 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_10
  simp only [← c63495, ← c63496, ← c63497] at e_1026_10
  have e_1026_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_11
  simp only [← c63498, ← c63499, k2, ← c63500] at e_1026_11
  have hr := e_1026_11
  simp only [e_1026_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8664)) := by
  have c63501 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63502 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63503 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63504 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63505 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63506 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_12
  simp only [← c63501, ← c63502, ← c63503] at e_1026_12
  have e_1026_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_13
  simp only [← c63504, ← c63505, k2, ← c63506] at e_1026_13
  have hr := e_1026_13
  simp only [e_1026_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8665)) := by
  have c63507 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63508 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63509 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63510 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63511 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63512 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_14
  simp only [← c63507, ← c63508, ← c63509] at e_1026_14
  have e_1026_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_15
  simp only [← c63510, ← c63511, k2, ← c63512] at e_1026_15
  have hr := e_1026_15
  simp only [e_1026_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8666)) := by
  have c63513 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63514 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63515 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63516 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63517 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63518 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_16
  simp only [← c63513, ← c63514, ← c63515] at e_1026_16
  have e_1026_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_17
  simp only [← c63516, ← c63517, k2, ← c63518] at e_1026_17
  have hr := e_1026_17
  simp only [e_1026_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1026 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8667)) := by
  have c63519 := (publicBatchWrapper86_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63520 := (publicBatchWrapper86_copies1985 a h).1
  have c63521 := (publicBatchWrapper86_copies1985 a h).2.1
  have c63522 := (publicBatchWrapper86_copies1985 a h).2.2.1
  have c63523 := (publicBatchWrapper86_copies1985 a h).2.2.2.1
  have c63524 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1026_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_18
  simp only [← c63519, ← c63520, ← c63521] at e_1026_18
  have e_1026_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_19
  simp only [← c63522, ← c63523, k2, ← c63524] at e_1026_19
  have hr := e_1026_19
  simp only [e_1026_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8668)) := by
  have c63525 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.1
  have c63526 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.1
  have c63527 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.1
  have c63528 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.1
  have c63529 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.1
  have c63530 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_0
  simp only [← c63525, ← c63526, ← c63527] at e_1027_0
  have e_1027_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_1
  simp only [← c63528, ← c63529, k2, ← c63530] at e_1027_1
  have hr := e_1027_1
  simp only [e_1027_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8669)) := by
  have c63531 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63532 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63533 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63534 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63535 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63536 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_2
  simp only [← c63531, ← c63532, ← c63533] at e_1027_2
  have e_1027_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_3
  simp only [← c63534, ← c63535, k2, ← c63536] at e_1027_3
  have hr := e_1027_3
  simp only [e_1027_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8670)) := by
  have c63537 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63538 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63539 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63540 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63541 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63542 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_4
  simp only [← c63537, ← c63538, ← c63539] at e_1027_4
  have e_1027_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_5
  simp only [← c63540, ← c63541, k2, ← c63542] at e_1027_5
  have hr := e_1027_5
  simp only [e_1027_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8671)) := by
  have c63543 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63544 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63545 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63546 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63547 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63548 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_6
  simp only [← c63543, ← c63544, ← c63545] at e_1027_6
  have e_1027_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_7
  simp only [← c63546, ← c63547, k2, ← c63548] at e_1027_7
  have hr := e_1027_7
  simp only [e_1027_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8672)) := by
  have c63549 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63550 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63551 := (publicBatchWrapper86_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63552 := (publicBatchWrapper86_copies1986 a h).1
  have c63553 := (publicBatchWrapper86_copies1986 a h).2.1
  have c63554 := (publicBatchWrapper86_copies1986 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_8
  simp only [← c63549, ← c63550, ← c63551] at e_1027_8
  have e_1027_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_9
  simp only [← c63552, ← c63553, k2, ← c63554] at e_1027_9
  have hr := e_1027_9
  simp only [e_1027_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8673)) := by
  have c63555 := (publicBatchWrapper86_copies1986 a h).2.2.2.1
  have c63556 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.1
  have c63557 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.1
  have c63558 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.1
  have c63559 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.1
  have c63560 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_10
  simp only [← c63555, ← c63556, ← c63557] at e_1027_10
  have e_1027_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_11
  simp only [← c63558, ← c63559, k2, ← c63560] at e_1027_11
  have hr := e_1027_11
  simp only [e_1027_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8674)) := by
  have c63561 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.1
  have c63562 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63563 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63564 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63565 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63566 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_12
  simp only [← c63561, ← c63562, ← c63563] at e_1027_12
  have e_1027_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_13
  simp only [← c63564, ← c63565, k2, ← c63566] at e_1027_13
  have hr := e_1027_13
  simp only [e_1027_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8675)) := by
  have c63567 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63568 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63569 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63570 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63571 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63572 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_14
  simp only [← c63567, ← c63568, ← c63569] at e_1027_14
  have e_1027_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_15
  simp only [← c63570, ← c63571, k2, ← c63572] at e_1027_15
  have hr := e_1027_15
  simp only [e_1027_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8676)) := by
  have c63573 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63574 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63575 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63576 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63577 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63578 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_16
  simp only [← c63573, ← c63574, ← c63575] at e_1027_16
  have e_1027_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_17
  simp only [← c63576, ← c63577, k2, ← c63578] at e_1027_17
  have hr := e_1027_17
  simp only [e_1027_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1027 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8677)) := by
  have c63579 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63580 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63581 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63582 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63583 := (publicBatchWrapper86_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63584 := (publicBatchWrapper86_copies1987 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1027_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_18
  simp only [← c63579, ← c63580, ← c63581] at e_1027_18
  have e_1027_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_19
  simp only [← c63582, ← c63583, k2, ← c63584] at e_1027_19
  have hr := e_1027_19
  simp only [e_1027_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8678)) := by
  have c63585 := (publicBatchWrapper86_copies1987 a h).2.1
  have c63586 := (publicBatchWrapper86_copies1987 a h).2.2.1
  have c63587 := (publicBatchWrapper86_copies1987 a h).2.2.2.1
  have c63588 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.1
  have c63589 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.1
  have c63590 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_0
  simp only [← c63585, ← c63586, ← c63587] at e_1028_0
  have e_1028_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_1
  simp only [← c63588, ← c63589, k2, ← c63590] at e_1028_1
  have hr := e_1028_1
  simp only [e_1028_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8679)) := by
  have c63591 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.1
  have c63592 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.1
  have c63593 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.1
  have c63594 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63595 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63596 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_2
  simp only [← c63591, ← c63592, ← c63593] at e_1028_2
  have e_1028_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_3
  simp only [← c63594, ← c63595, k2, ← c63596] at e_1028_3
  have hr := e_1028_3
  simp only [e_1028_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8680)) := by
  have c63597 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63598 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63599 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63600 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63601 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63602 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_4
  simp only [← c63597, ← c63598, ← c63599] at e_1028_4
  have e_1028_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_5
  simp only [← c63600, ← c63601, k2, ← c63602] at e_1028_5
  have hr := e_1028_5
  simp only [e_1028_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8681)) := by
  have c63603 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63604 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63605 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63606 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63607 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63608 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_6
  simp only [← c63603, ← c63604, ← c63605] at e_1028_6
  have e_1028_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_7
  simp only [← c63606, ← c63607, k2, ← c63608] at e_1028_7
  have hr := e_1028_7
  simp only [e_1028_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8682)) := by
  have c63609 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63610 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63611 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63612 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63613 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63614 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_8
  simp only [← c63609, ← c63610, ← c63611] at e_1028_8
  have e_1028_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_9
  simp only [← c63612, ← c63613, k2, ← c63614] at e_1028_9
  have hr := e_1028_9
  simp only [e_1028_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8683)) := by
  have c63615 := (publicBatchWrapper86_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63616 := (publicBatchWrapper86_copies1988 a h).1
  have c63617 := (publicBatchWrapper86_copies1988 a h).2.1
  have c63618 := (publicBatchWrapper86_copies1988 a h).2.2.1
  have c63619 := (publicBatchWrapper86_copies1988 a h).2.2.2.1
  have c63620 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_10
  simp only [← c63615, ← c63616, ← c63617] at e_1028_10
  have e_1028_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_11
  simp only [← c63618, ← c63619, k2, ← c63620] at e_1028_11
  have hr := e_1028_11
  simp only [e_1028_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8684)) := by
  have c63621 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.1
  have c63622 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.1
  have c63623 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.1
  have c63624 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.1
  have c63625 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.1
  have c63626 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_12
  simp only [← c63621, ← c63622, ← c63623] at e_1028_12
  have e_1028_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_13
  simp only [← c63624, ← c63625, k2, ← c63626] at e_1028_13
  have hr := e_1028_13
  simp only [e_1028_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8685)) := by
  have c63627 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63628 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63629 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63630 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63631 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63632 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_14
  simp only [← c63627, ← c63628, ← c63629] at e_1028_14
  have e_1028_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_15
  simp only [← c63630, ← c63631, k2, ← c63632] at e_1028_15
  have hr := e_1028_15
  simp only [e_1028_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8686)) := by
  have c63633 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63634 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63635 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63636 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63637 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63638 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_16
  simp only [← c63633, ← c63634, ← c63635] at e_1028_16
  have e_1028_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_17
  simp only [← c63636, ← c63637, k2, ← c63638] at e_1028_17
  have hr := e_1028_17
  simp only [e_1028_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1028 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8687)) := by
  have c63639 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63640 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63641 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63642 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63643 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63644 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1028_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_18
  simp only [← c63639, ← c63640, ← c63641] at e_1028_18
  have e_1028_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_19
  simp only [← c63642, ← c63643, k2, ← c63644] at e_1028_19
  have hr := e_1028_19
  simp only [e_1028_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8688)) := by
  have c63645 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63646 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63647 := (publicBatchWrapper86_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63648 := (publicBatchWrapper86_copies1989 a h).1
  have c63649 := (publicBatchWrapper86_copies1989 a h).2.1
  have c63650 := (publicBatchWrapper86_copies1989 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_0
  simp only [← c63645, ← c63646, ← c63647] at e_1029_0
  have e_1029_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_1
  simp only [← c63648, ← c63649, k2, ← c63650] at e_1029_1
  have hr := e_1029_1
  simp only [e_1029_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8689)) := by
  have c63651 := (publicBatchWrapper86_copies1989 a h).2.2.2.1
  have c63652 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.1
  have c63653 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.1
  have c63654 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.1
  have c63655 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.1
  have c63656 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_2
  simp only [← c63651, ← c63652, ← c63653] at e_1029_2
  have e_1029_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_3
  simp only [← c63654, ← c63655, k2, ← c63656] at e_1029_3
  have hr := e_1029_3
  simp only [e_1029_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8690)) := by
  have c63657 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.1
  have c63658 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63659 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63660 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63661 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63662 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_4
  simp only [← c63657, ← c63658, ← c63659] at e_1029_4
  have e_1029_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_5
  simp only [← c63660, ← c63661, k2, ← c63662] at e_1029_5
  have hr := e_1029_5
  simp only [e_1029_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8691)) := by
  have c63663 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63664 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63665 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63666 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63667 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63668 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_6
  simp only [← c63663, ← c63664, ← c63665] at e_1029_6
  have e_1029_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_7
  simp only [← c63666, ← c63667, k2, ← c63668] at e_1029_7
  have hr := e_1029_7
  simp only [e_1029_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8692)) := by
  have c63669 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63670 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63671 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63672 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63673 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63674 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_8
  simp only [← c63669, ← c63670, ← c63671] at e_1029_8
  have e_1029_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_9
  simp only [← c63672, ← c63673, k2, ← c63674] at e_1029_9
  have hr := e_1029_9
  simp only [e_1029_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8693)) := by
  have c63675 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63676 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63677 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63678 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63679 := (publicBatchWrapper86_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63680 := (publicBatchWrapper86_copies1990 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_10
  simp only [← c63675, ← c63676, ← c63677] at e_1029_10
  have e_1029_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_11
  simp only [← c63678, ← c63679, k2, ← c63680] at e_1029_11
  have hr := e_1029_11
  simp only [e_1029_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8694)) := by
  have c63681 := (publicBatchWrapper86_copies1990 a h).2.1
  have c63682 := (publicBatchWrapper86_copies1990 a h).2.2.1
  have c63683 := (publicBatchWrapper86_copies1990 a h).2.2.2.1
  have c63684 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.1
  have c63685 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.1
  have c63686 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_12
  simp only [← c63681, ← c63682, ← c63683] at e_1029_12
  have e_1029_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_13
  simp only [← c63684, ← c63685, k2, ← c63686] at e_1029_13
  have hr := e_1029_13
  simp only [e_1029_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8695)) := by
  have c63687 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.1
  have c63688 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.1
  have c63689 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.1
  have c63690 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63691 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63692 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_14
  simp only [← c63687, ← c63688, ← c63689] at e_1029_14
  have e_1029_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_15
  simp only [← c63690, ← c63691, k2, ← c63692] at e_1029_15
  have hr := e_1029_15
  simp only [e_1029_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8696)) := by
  have c63693 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63694 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63695 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63696 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63697 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63698 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_16
  simp only [← c63693, ← c63694, ← c63695] at e_1029_16
  have e_1029_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_17
  simp only [← c63696, ← c63697, k2, ← c63698] at e_1029_17
  have hr := e_1029_17
  simp only [e_1029_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1029 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8697)) := by
  have c63699 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63700 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63701 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63702 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63703 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63704 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1029_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_18
  simp only [← c63699, ← c63700, ← c63701] at e_1029_18
  have e_1029_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_19
  simp only [← c63702, ← c63703, k2, ← c63704] at e_1029_19
  have hr := e_1029_19
  simp only [e_1029_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8698)) := by
  have c63705 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63706 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63707 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63708 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63709 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63710 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_0
  simp only [← c63705, ← c63706, ← c63707] at e_1030_0
  have e_1030_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_1
  simp only [← c63708, ← c63709, k2, ← c63710] at e_1030_1
  have hr := e_1030_1
  simp only [e_1030_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8699)) := by
  have c63711 := (publicBatchWrapper86_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63712 := (publicBatchWrapper86_copies1991 a h).1
  have c63713 := (publicBatchWrapper86_copies1991 a h).2.1
  have c63714 := (publicBatchWrapper86_copies1991 a h).2.2.1
  have c63715 := (publicBatchWrapper86_copies1991 a h).2.2.2.1
  have c63716 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_2
  simp only [← c63711, ← c63712, ← c63713] at e_1030_2
  have e_1030_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_3
  simp only [← c63714, ← c63715, k2, ← c63716] at e_1030_3
  have hr := e_1030_3
  simp only [e_1030_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8700)) := by
  have c63717 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.1
  have c63718 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.1
  have c63719 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.1
  have c63720 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.1
  have c63721 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.1
  have c63722 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_4
  simp only [← c63717, ← c63718, ← c63719] at e_1030_4
  have e_1030_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_5
  simp only [← c63720, ← c63721, k2, ← c63722] at e_1030_5
  have hr := e_1030_5
  simp only [e_1030_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8701)) := by
  have c63723 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63724 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63725 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63726 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63727 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63728 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_6
  simp only [← c63723, ← c63724, ← c63725] at e_1030_6
  have e_1030_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_7
  simp only [← c63726, ← c63727, k2, ← c63728] at e_1030_7
  have hr := e_1030_7
  simp only [e_1030_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8702)) := by
  have c63729 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63730 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63731 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63732 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63733 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63734 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_8
  simp only [← c63729, ← c63730, ← c63731] at e_1030_8
  have e_1030_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_9
  simp only [← c63732, ← c63733, k2, ← c63734] at e_1030_9
  have hr := e_1030_9
  simp only [e_1030_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8703)) := by
  have c63735 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63736 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63737 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63738 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63739 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63740 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_10
  simp only [← c63735, ← c63736, ← c63737] at e_1030_10
  have e_1030_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_11
  simp only [← c63738, ← c63739, k2, ← c63740] at e_1030_11
  have hr := e_1030_11
  simp only [e_1030_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8704)) := by
  have c63741 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63742 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63743 := (publicBatchWrapper86_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63744 := (publicBatchWrapper86_copies1992 a h).1
  have c63745 := (publicBatchWrapper86_copies1992 a h).2.1
  have c63746 := (publicBatchWrapper86_copies1992 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_12
  simp only [← c63741, ← c63742, ← c63743] at e_1030_12
  have e_1030_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_13
  simp only [← c63744, ← c63745, k2, ← c63746] at e_1030_13
  have hr := e_1030_13
  simp only [e_1030_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8705)) := by
  have c63747 := (publicBatchWrapper86_copies1992 a h).2.2.2.1
  have c63748 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.1
  have c63749 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.1
  have c63750 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.1
  have c63751 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.1
  have c63752 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_14
  simp only [← c63747, ← c63748, ← c63749] at e_1030_14
  have e_1030_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_15
  simp only [← c63750, ← c63751, k2, ← c63752] at e_1030_15
  have hr := e_1030_15
  simp only [e_1030_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8706)) := by
  have c63753 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.1
  have c63754 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63755 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63756 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63757 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63758 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_16
  simp only [← c63753, ← c63754, ← c63755] at e_1030_16
  have e_1030_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_17
  simp only [← c63756, ← c63757, k2, ← c63758] at e_1030_17
  have hr := e_1030_17
  simp only [e_1030_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1030 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8707)) := by
  have c63759 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63760 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63761 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63762 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63763 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63764 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1030_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_18
  simp only [← c63759, ← c63760, ← c63761] at e_1030_18
  have e_1030_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_19
  simp only [← c63762, ← c63763, k2, ← c63764] at e_1030_19
  have hr := e_1030_19
  simp only [e_1030_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8708)) := by
  have c63765 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63766 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63767 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63768 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63769 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63770 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_0
  simp only [← c63765, ← c63766, ← c63767] at e_1031_0
  have e_1031_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_1
  simp only [← c63768, ← c63769, k2, ← c63770] at e_1031_1
  have hr := e_1031_1
  simp only [e_1031_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8709)) := by
  have c63771 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63772 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63773 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63774 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63775 := (publicBatchWrapper86_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63776 := (publicBatchWrapper86_copies1993 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_2
  simp only [← c63771, ← c63772, ← c63773] at e_1031_2
  have e_1031_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_3
  simp only [← c63774, ← c63775, k2, ← c63776] at e_1031_3
  have hr := e_1031_3
  simp only [e_1031_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8710)) := by
  have c63777 := (publicBatchWrapper86_copies1993 a h).2.1
  have c63778 := (publicBatchWrapper86_copies1993 a h).2.2.1
  have c63779 := (publicBatchWrapper86_copies1993 a h).2.2.2.1
  have c63780 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.1
  have c63781 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.1
  have c63782 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_4
  simp only [← c63777, ← c63778, ← c63779] at e_1031_4
  have e_1031_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_5
  simp only [← c63780, ← c63781, k2, ← c63782] at e_1031_5
  have hr := e_1031_5
  simp only [e_1031_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8711)) := by
  have c63783 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.1
  have c63784 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.1
  have c63785 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.1
  have c63786 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63787 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63788 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_6
  simp only [← c63783, ← c63784, ← c63785] at e_1031_6
  have e_1031_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_7
  simp only [← c63786, ← c63787, k2, ← c63788] at e_1031_7
  have hr := e_1031_7
  simp only [e_1031_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8712)) := by
  have c63789 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63790 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63791 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63792 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63793 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63794 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_8
  simp only [← c63789, ← c63790, ← c63791] at e_1031_8
  have e_1031_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_9
  simp only [← c63792, ← c63793, k2, ← c63794] at e_1031_9
  have hr := e_1031_9
  simp only [e_1031_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8713)) := by
  have c63795 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63796 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63797 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63798 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63799 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63800 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_10
  simp only [← c63795, ← c63796, ← c63797] at e_1031_10
  have e_1031_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_11
  simp only [← c63798, ← c63799, k2, ← c63800] at e_1031_11
  have hr := e_1031_11
  simp only [e_1031_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8714)) := by
  have c63801 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63802 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63803 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63804 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63805 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63806 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_12
  simp only [← c63801, ← c63802, ← c63803] at e_1031_12
  have e_1031_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_13
  simp only [← c63804, ← c63805, k2, ← c63806] at e_1031_13
  have hr := e_1031_13
  simp only [e_1031_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8715)) := by
  have c63807 := (publicBatchWrapper86_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63808 := (publicBatchWrapper86_copies1994 a h).1
  have c63809 := (publicBatchWrapper86_copies1994 a h).2.1
  have c63810 := (publicBatchWrapper86_copies1994 a h).2.2.1
  have c63811 := (publicBatchWrapper86_copies1994 a h).2.2.2.1
  have c63812 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_14
  simp only [← c63807, ← c63808, ← c63809] at e_1031_14
  have e_1031_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_15
  simp only [← c63810, ← c63811, k2, ← c63812] at e_1031_15
  have hr := e_1031_15
  simp only [e_1031_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8716)) := by
  have c63813 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.1
  have c63814 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.1
  have c63815 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.1
  have c63816 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.1
  have c63817 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.1
  have c63818 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_16
  simp only [← c63813, ← c63814, ← c63815] at e_1031_16
  have e_1031_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_17
  simp only [← c63816, ← c63817, k2, ← c63818] at e_1031_17
  have hr := e_1031_17
  simp only [e_1031_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1031 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8717)) := by
  have c63819 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63820 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63821 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63822 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63823 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63824 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1031_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_18
  simp only [← c63819, ← c63820, ← c63821] at e_1031_18
  have e_1031_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_19
  simp only [← c63822, ← c63823, k2, ← c63824] at e_1031_19
  have hr := e_1031_19
  simp only [e_1031_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8718)) := by
  have c63825 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63826 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63827 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63828 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63829 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63830 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_0
  simp only [← c63825, ← c63826, ← c63827] at e_1032_0
  have e_1032_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_1
  simp only [← c63828, ← c63829, k2, ← c63830] at e_1032_1
  have hr := e_1032_1
  simp only [e_1032_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8719)) := by
  have c63831 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63832 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63833 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63834 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63835 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63836 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_2
  simp only [← c63831, ← c63832, ← c63833] at e_1032_2
  have e_1032_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_3
  simp only [← c63834, ← c63835, k2, ← c63836] at e_1032_3
  have hr := e_1032_3
  simp only [e_1032_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8720)) := by
  have c63837 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63838 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63839 := (publicBatchWrapper86_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63840 := (publicBatchWrapper86_copies1995 a h).1
  have c63841 := (publicBatchWrapper86_copies1995 a h).2.1
  have c63842 := (publicBatchWrapper86_copies1995 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_4
  simp only [← c63837, ← c63838, ← c63839] at e_1032_4
  have e_1032_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_5
  simp only [← c63840, ← c63841, k2, ← c63842] at e_1032_5
  have hr := e_1032_5
  simp only [e_1032_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8721)) := by
  have c63843 := (publicBatchWrapper86_copies1995 a h).2.2.2.1
  have c63844 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.1
  have c63845 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.1
  have c63846 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.1
  have c63847 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.1
  have c63848 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_6
  simp only [← c63843, ← c63844, ← c63845] at e_1032_6
  have e_1032_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_7
  simp only [← c63846, ← c63847, k2, ← c63848] at e_1032_7
  have hr := e_1032_7
  simp only [e_1032_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8722)) := by
  have c63849 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.1
  have c63850 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63851 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63852 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63853 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63854 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_8
  simp only [← c63849, ← c63850, ← c63851] at e_1032_8
  have e_1032_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_9
  simp only [← c63852, ← c63853, k2, ← c63854] at e_1032_9
  have hr := e_1032_9
  simp only [e_1032_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8723)) := by
  have c63855 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63856 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63857 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63858 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63859 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63860 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_10
  simp only [← c63855, ← c63856, ← c63857] at e_1032_10
  have e_1032_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_11
  simp only [← c63858, ← c63859, k2, ← c63860] at e_1032_11
  have hr := e_1032_11
  simp only [e_1032_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8724)) := by
  have c63861 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63862 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63863 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63864 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63865 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63866 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_12
  simp only [← c63861, ← c63862, ← c63863] at e_1032_12
  have e_1032_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_13
  simp only [← c63864, ← c63865, k2, ← c63866] at e_1032_13
  have hr := e_1032_13
  simp only [e_1032_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8725)) := by
  have c63867 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63868 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63869 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63870 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63871 := (publicBatchWrapper86_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63872 := (publicBatchWrapper86_copies1996 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_14
  simp only [← c63867, ← c63868, ← c63869] at e_1032_14
  have e_1032_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_15
  simp only [← c63870, ← c63871, k2, ← c63872] at e_1032_15
  have hr := e_1032_15
  simp only [e_1032_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8726)) := by
  have c63873 := (publicBatchWrapper86_copies1996 a h).2.1
  have c63874 := (publicBatchWrapper86_copies1996 a h).2.2.1
  have c63875 := (publicBatchWrapper86_copies1996 a h).2.2.2.1
  have c63876 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.1
  have c63877 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.1
  have c63878 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_16
  simp only [← c63873, ← c63874, ← c63875] at e_1032_16
  have e_1032_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_17
  simp only [← c63876, ← c63877, k2, ← c63878] at e_1032_17
  have hr := e_1032_17
  simp only [e_1032_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1032 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8727)) := by
  have c63879 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.1
  have c63880 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.1
  have c63881 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.1
  have c63882 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63883 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63884 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1032_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_18
  simp only [← c63879, ← c63880, ← c63881] at e_1032_18
  have e_1032_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_19
  simp only [← c63882, ← c63883, k2, ← c63884] at e_1032_19
  have hr := e_1032_19
  simp only [e_1032_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8728)) := by
  have c63885 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63886 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63887 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63888 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63889 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63890 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_0
  simp only [← c63885, ← c63886, ← c63887] at e_1033_0
  have e_1033_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_1
  simp only [← c63888, ← c63889, k2, ← c63890] at e_1033_1
  have hr := e_1033_1
  simp only [e_1033_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8729)) := by
  have c63891 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63892 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63893 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63894 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63895 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63896 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_2
  simp only [← c63891, ← c63892, ← c63893] at e_1033_2
  have e_1033_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_3
  simp only [← c63894, ← c63895, k2, ← c63896] at e_1033_3
  have hr := e_1033_3
  simp only [e_1033_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8730)) := by
  have c63897 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63898 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63899 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63900 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63901 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63902 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_4
  simp only [← c63897, ← c63898, ← c63899] at e_1033_4
  have e_1033_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_5
  simp only [← c63900, ← c63901, k2, ← c63902] at e_1033_5
  have hr := e_1033_5
  simp only [e_1033_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8731)) := by
  have c63903 := (publicBatchWrapper86_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63904 := (publicBatchWrapper86_copies1997 a h).1
  have c63905 := (publicBatchWrapper86_copies1997 a h).2.1
  have c63906 := (publicBatchWrapper86_copies1997 a h).2.2.1
  have c63907 := (publicBatchWrapper86_copies1997 a h).2.2.2.1
  have c63908 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_6
  simp only [← c63903, ← c63904, ← c63905] at e_1033_6
  have e_1033_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_7
  simp only [← c63906, ← c63907, k2, ← c63908] at e_1033_7
  have hr := e_1033_7
  simp only [e_1033_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8732)) := by
  have c63909 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.1
  have c63910 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.1
  have c63911 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.1
  have c63912 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.1
  have c63913 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.1
  have c63914 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_8
  simp only [← c63909, ← c63910, ← c63911] at e_1033_8
  have e_1033_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_9
  simp only [← c63912, ← c63913, k2, ← c63914] at e_1033_9
  have hr := e_1033_9
  simp only [e_1033_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8733)) := by
  have c63915 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63916 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63917 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63918 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63919 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63920 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_10
  simp only [← c63915, ← c63916, ← c63917] at e_1033_10
  have e_1033_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_11
  simp only [← c63918, ← c63919, k2, ← c63920] at e_1033_11
  have hr := e_1033_11
  simp only [e_1033_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8734)) := by
  have c63921 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63922 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63923 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63924 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63925 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63926 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_12
  simp only [← c63921, ← c63922, ← c63923] at e_1033_12
  have e_1033_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_13
  simp only [← c63924, ← c63925, k2, ← c63926] at e_1033_13
  have hr := e_1033_13
  simp only [e_1033_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8735)) := by
  have c63927 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63928 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63929 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63930 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63931 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63932 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_14
  simp only [← c63927, ← c63928, ← c63929] at e_1033_14
  have e_1033_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_15
  simp only [← c63930, ← c63931, k2, ← c63932] at e_1033_15
  have hr := e_1033_15
  simp only [e_1033_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8736)) := by
  have c63933 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63934 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63935 := (publicBatchWrapper86_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63936 := (publicBatchWrapper86_copies1998 a h).1
  have c63937 := (publicBatchWrapper86_copies1998 a h).2.1
  have c63938 := (publicBatchWrapper86_copies1998 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_16
  simp only [← c63933, ← c63934, ← c63935] at e_1033_16
  have e_1033_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_17
  simp only [← c63936, ← c63937, k2, ← c63938] at e_1033_17
  have hr := e_1033_17
  simp only [e_1033_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1033 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8737)) := by
  have c63939 := (publicBatchWrapper86_copies1998 a h).2.2.2.1
  have c63940 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.1
  have c63941 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.1
  have c63942 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.1
  have c63943 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.1
  have c63944 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1033_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_18
  simp only [← c63939, ← c63940, ← c63941] at e_1033_18
  have e_1033_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_19
  simp only [← c63942, ← c63943, k2, ← c63944] at e_1033_19
  have hr := e_1033_19
  simp only [e_1033_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8738)) := by
  have c63945 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.1
  have c63946 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63947 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63948 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63949 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63950 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_0
  simp only [← c63945, ← c63946, ← c63947] at e_1034_0
  have e_1034_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_1
  simp only [← c63948, ← c63949, k2, ← c63950] at e_1034_1
  have hr := e_1034_1
  simp only [e_1034_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8739)) := by
  have c63951 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63952 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63953 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63954 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63955 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63956 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_2
  simp only [← c63951, ← c63952, ← c63953] at e_1034_2
  have e_1034_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_3
  simp only [← c63954, ← c63955, k2, ← c63956] at e_1034_3
  have hr := e_1034_3
  simp only [e_1034_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8740)) := by
  have c63957 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63958 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63959 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63960 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63961 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63962 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_4
  simp only [← c63957, ← c63958, ← c63959] at e_1034_4
  have e_1034_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_5
  simp only [← c63960, ← c63961, k2, ← c63962] at e_1034_5
  have hr := e_1034_5
  simp only [e_1034_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8741)) := by
  have c63963 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63964 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63965 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63966 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63967 := (publicBatchWrapper86_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63968 := (publicBatchWrapper86_copies1999 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_6
  simp only [← c63963, ← c63964, ← c63965] at e_1034_6
  have e_1034_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_7
  simp only [← c63966, ← c63967, k2, ← c63968] at e_1034_7
  have hr := e_1034_7
  simp only [e_1034_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8742)) := by
  have c63969 := (publicBatchWrapper86_copies1999 a h).2.1
  have c63970 := (publicBatchWrapper86_copies1999 a h).2.2.1
  have c63971 := (publicBatchWrapper86_copies1999 a h).2.2.2.1
  have c63972 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.1
  have c63973 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.1
  have c63974 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_8
  simp only [← c63969, ← c63970, ← c63971] at e_1034_8
  have e_1034_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_9
  simp only [← c63972, ← c63973, k2, ← c63974] at e_1034_9
  have hr := e_1034_9
  simp only [e_1034_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8743)) := by
  have c63975 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.1
  have c63976 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.1
  have c63977 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.1
  have c63978 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63979 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63980 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_10
  simp only [← c63975, ← c63976, ← c63977] at e_1034_10
  have e_1034_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_11
  simp only [← c63978, ← c63979, k2, ← c63980] at e_1034_11
  have hr := e_1034_11
  simp only [e_1034_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8744)) := by
  have c63981 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63982 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63983 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63984 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63985 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63986 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_12
  simp only [← c63981, ← c63982, ← c63983] at e_1034_12
  have e_1034_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_13
  simp only [← c63984, ← c63985, k2, ← c63986] at e_1034_13
  have hr := e_1034_13
  simp only [e_1034_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8745)) := by
  have c63987 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63988 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63989 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63990 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63991 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63992 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_14
  simp only [← c63987, ← c63988, ← c63989] at e_1034_14
  have e_1034_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_15
  simp only [← c63990, ← c63991, k2, ← c63992] at e_1034_15
  have hr := e_1034_15
  simp only [e_1034_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8746)) := by
  have c63993 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63994 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63995 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63996 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63997 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63998 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_16
  simp only [← c63993, ← c63994, ← c63995] at e_1034_16
  have e_1034_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_17
  simp only [← c63996, ← c63997, k2, ← c63998] at e_1034_17
  have hr := e_1034_17
  simp only [e_1034_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1034 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8747)) := by
  have c63999 := (publicBatchWrapper86_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64000 := (publicBatchWrapper86_copies2000 a h).1
  have c64001 := (publicBatchWrapper86_copies2000 a h).2.1
  have c64002 := (publicBatchWrapper86_copies2000 a h).2.2.1
  have c64003 := (publicBatchWrapper86_copies2000 a h).2.2.2.1
  have c64004 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1034_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_18
  simp only [← c63999, ← c64000, ← c64001] at e_1034_18
  have e_1034_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_19
  simp only [← c64002, ← c64003, k2, ← c64004] at e_1034_19
  have hr := e_1034_19
  simp only [e_1034_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8748)) := by
  have c64005 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.1
  have c64006 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.1
  have c64007 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.1
  have c64008 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.1
  have c64009 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.1
  have c64010 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_0
  simp only [← c64005, ← c64006, ← c64007] at e_1035_0
  have e_1035_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_1
  simp only [← c64008, ← c64009, k2, ← c64010] at e_1035_1
  have hr := e_1035_1
  simp only [e_1035_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8749)) := by
  have c64011 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64012 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64013 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64014 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64015 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64016 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_2
  simp only [← c64011, ← c64012, ← c64013] at e_1035_2
  have e_1035_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_3
  simp only [← c64014, ← c64015, k2, ← c64016] at e_1035_3
  have hr := e_1035_3
  simp only [e_1035_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8750)) := by
  have c64017 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64018 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64019 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64020 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64021 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64022 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_4
  simp only [← c64017, ← c64018, ← c64019] at e_1035_4
  have e_1035_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_5
  simp only [← c64020, ← c64021, k2, ← c64022] at e_1035_5
  have hr := e_1035_5
  simp only [e_1035_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8751)) := by
  have c64023 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64024 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64025 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64026 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64027 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64028 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_6
  simp only [← c64023, ← c64024, ← c64025] at e_1035_6
  have e_1035_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_7
  simp only [← c64026, ← c64027, k2, ← c64028] at e_1035_7
  have hr := e_1035_7
  simp only [e_1035_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8752)) := by
  have c64029 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64030 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64031 := (publicBatchWrapper86_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64032 := (publicBatchWrapper86_copies2001 a h).1
  have c64033 := (publicBatchWrapper86_copies2001 a h).2.1
  have c64034 := (publicBatchWrapper86_copies2001 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_8
  simp only [← c64029, ← c64030, ← c64031] at e_1035_8
  have e_1035_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_9
  simp only [← c64032, ← c64033, k2, ← c64034] at e_1035_9
  have hr := e_1035_9
  simp only [e_1035_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8753)) := by
  have c64035 := (publicBatchWrapper86_copies2001 a h).2.2.2.1
  have c64036 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.1
  have c64037 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.1
  have c64038 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.1
  have c64039 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.1
  have c64040 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_10
  simp only [← c64035, ← c64036, ← c64037] at e_1035_10
  have e_1035_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_11
  simp only [← c64038, ← c64039, k2, ← c64040] at e_1035_11
  have hr := e_1035_11
  simp only [e_1035_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8754)) := by
  have c64041 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.1
  have c64042 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64043 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64044 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64045 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64046 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_12
  simp only [← c64041, ← c64042, ← c64043] at e_1035_12
  have e_1035_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_13
  simp only [← c64044, ← c64045, k2, ← c64046] at e_1035_13
  have hr := e_1035_13
  simp only [e_1035_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8755)) := by
  have c64047 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64048 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64049 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64050 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64051 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64052 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_14
  simp only [← c64047, ← c64048, ← c64049] at e_1035_14
  have e_1035_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_15
  simp only [← c64050, ← c64051, k2, ← c64052] at e_1035_15
  have hr := e_1035_15
  simp only [e_1035_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8756)) := by
  have c64053 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64054 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64055 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64056 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64057 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64058 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_16
  simp only [← c64053, ← c64054, ← c64055] at e_1035_16
  have e_1035_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_17
  simp only [← c64056, ← c64057, k2, ← c64058] at e_1035_17
  have hr := e_1035_17
  simp only [e_1035_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1035 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8757)) := by
  have c64059 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64060 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64061 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64062 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64063 := (publicBatchWrapper86_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64064 := (publicBatchWrapper86_copies2002 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1035_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_18
  simp only [← c64059, ← c64060, ← c64061] at e_1035_18
  have e_1035_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_19
  simp only [← c64062, ← c64063, k2, ← c64064] at e_1035_19
  have hr := e_1035_19
  simp only [e_1035_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8758)) := by
  have c64065 := (publicBatchWrapper86_copies2002 a h).2.1
  have c64066 := (publicBatchWrapper86_copies2002 a h).2.2.1
  have c64067 := (publicBatchWrapper86_copies2002 a h).2.2.2.1
  have c64068 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.1
  have c64069 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.1
  have c64070 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_0
  simp only [← c64065, ← c64066, ← c64067] at e_1036_0
  have e_1036_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_1
  simp only [← c64068, ← c64069, k2, ← c64070] at e_1036_1
  have hr := e_1036_1
  simp only [e_1036_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8759)) := by
  have c64071 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.1
  have c64072 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.1
  have c64073 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.1
  have c64074 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64075 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64076 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_2
  simp only [← c64071, ← c64072, ← c64073] at e_1036_2
  have e_1036_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_3
  simp only [← c64074, ← c64075, k2, ← c64076] at e_1036_3
  have hr := e_1036_3
  simp only [e_1036_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8760)) := by
  have c64077 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64078 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64079 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64080 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64081 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64082 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_4
  simp only [← c64077, ← c64078, ← c64079] at e_1036_4
  have e_1036_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_5
  simp only [← c64080, ← c64081, k2, ← c64082] at e_1036_5
  have hr := e_1036_5
  simp only [e_1036_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8761)) := by
  have c64083 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64084 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64085 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64086 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64087 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64088 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_6
  simp only [← c64083, ← c64084, ← c64085] at e_1036_6
  have e_1036_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_7
  simp only [← c64086, ← c64087, k2, ← c64088] at e_1036_7
  have hr := e_1036_7
  simp only [e_1036_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8762)) := by
  have c64089 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64090 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64091 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64092 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64093 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64094 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_8
  simp only [← c64089, ← c64090, ← c64091] at e_1036_8
  have e_1036_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_9
  simp only [← c64092, ← c64093, k2, ← c64094] at e_1036_9
  have hr := e_1036_9
  simp only [e_1036_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8763)) := by
  have c64095 := (publicBatchWrapper86_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64096 := (publicBatchWrapper86_copies2003 a h).1
  have c64097 := (publicBatchWrapper86_copies2003 a h).2.1
  have c64098 := (publicBatchWrapper86_copies2003 a h).2.2.1
  have c64099 := (publicBatchWrapper86_copies2003 a h).2.2.2.1
  have c64100 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_10
  simp only [← c64095, ← c64096, ← c64097] at e_1036_10
  have e_1036_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_11
  simp only [← c64098, ← c64099, k2, ← c64100] at e_1036_11
  have hr := e_1036_11
  simp only [e_1036_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8764)) := by
  have c64101 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.1
  have c64102 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.1
  have c64103 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.1
  have c64104 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.1
  have c64105 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.1
  have c64106 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_12
  simp only [← c64101, ← c64102, ← c64103] at e_1036_12
  have e_1036_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_13
  simp only [← c64104, ← c64105, k2, ← c64106] at e_1036_13
  have hr := e_1036_13
  simp only [e_1036_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8765)) := by
  have c64107 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64108 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64109 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64110 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64111 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64112 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_14
  simp only [← c64107, ← c64108, ← c64109] at e_1036_14
  have e_1036_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_15
  simp only [← c64110, ← c64111, k2, ← c64112] at e_1036_15
  have hr := e_1036_15
  simp only [e_1036_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8766)) := by
  have c64113 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64114 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64115 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64116 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64117 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64118 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_16
  simp only [← c64113, ← c64114, ← c64115] at e_1036_16
  have e_1036_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_17
  simp only [← c64116, ← c64117, k2, ← c64118] at e_1036_17
  have hr := e_1036_17
  simp only [e_1036_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1036 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8767)) := by
  have c64119 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64120 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64121 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64122 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64123 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64124 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1036_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_18
  simp only [← c64119, ← c64120, ← c64121] at e_1036_18
  have e_1036_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_19
  simp only [← c64122, ← c64123, k2, ← c64124] at e_1036_19
  have hr := e_1036_19
  simp only [e_1036_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8768)) := by
  have c64125 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64126 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64127 := (publicBatchWrapper86_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64128 := (publicBatchWrapper86_copies2004 a h).1
  have c64129 := (publicBatchWrapper86_copies2004 a h).2.1
  have c64130 := (publicBatchWrapper86_copies2004 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_0
  simp only [← c64125, ← c64126, ← c64127] at e_1037_0
  have e_1037_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_1
  simp only [← c64128, ← c64129, k2, ← c64130] at e_1037_1
  have hr := e_1037_1
  simp only [e_1037_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8769)) := by
  have c64131 := (publicBatchWrapper86_copies2004 a h).2.2.2.1
  have c64132 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.1
  have c64133 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.1
  have c64134 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.1
  have c64135 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.1
  have c64136 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_2
  simp only [← c64131, ← c64132, ← c64133] at e_1037_2
  have e_1037_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_3
  simp only [← c64134, ← c64135, k2, ← c64136] at e_1037_3
  have hr := e_1037_3
  simp only [e_1037_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8810)) := by
  have c64137 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.1
  have c64138 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64139 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64140 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64141 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64142 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_4
  simp only [← c64137, ← c64138, ← c64139] at e_1037_4
  have e_1037_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_5
  simp only [← c64140, ← c64141, k2, ← c64142] at e_1037_5
  have hr := e_1037_5
  simp only [e_1037_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8811)) := by
  have c64143 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64144 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64145 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64146 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64147 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64148 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_6
  simp only [← c64143, ← c64144, ← c64145] at e_1037_6
  have e_1037_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_7
  simp only [← c64146, ← c64147, k2, ← c64148] at e_1037_7
  have hr := e_1037_7
  simp only [e_1037_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8812)) := by
  have c64149 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64150 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64151 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64152 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64153 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64154 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_8
  simp only [← c64149, ← c64150, ← c64151] at e_1037_8
  have e_1037_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_9
  simp only [← c64152, ← c64153, k2, ← c64154] at e_1037_9
  have hr := e_1037_9
  simp only [e_1037_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8813)) := by
  have c64155 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64156 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64157 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64158 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64159 := (publicBatchWrapper86_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64160 := (publicBatchWrapper86_copies2005 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_10
  simp only [← c64155, ← c64156, ← c64157] at e_1037_10
  have e_1037_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_11
  simp only [← c64158, ← c64159, k2, ← c64160] at e_1037_11
  have hr := e_1037_11
  simp only [e_1037_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8814)) := by
  have c64161 := (publicBatchWrapper86_copies2005 a h).2.1
  have c64162 := (publicBatchWrapper86_copies2005 a h).2.2.1
  have c64163 := (publicBatchWrapper86_copies2005 a h).2.2.2.1
  have c64164 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.1
  have c64165 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.1
  have c64166 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_12
  simp only [← c64161, ← c64162, ← c64163] at e_1037_12
  have e_1037_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_13
  simp only [← c64164, ← c64165, k2, ← c64166] at e_1037_13
  have hr := e_1037_13
  simp only [e_1037_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8815)) := by
  have c64167 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.1
  have c64168 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.1
  have c64169 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.1
  have c64170 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64171 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64172 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_14
  simp only [← c64167, ← c64168, ← c64169] at e_1037_14
  have e_1037_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_15
  simp only [← c64170, ← c64171, k2, ← c64172] at e_1037_15
  have hr := e_1037_15
  simp only [e_1037_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8816)) := by
  have c64173 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64174 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64175 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64176 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64177 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64178 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_16
  simp only [← c64173, ← c64174, ← c64175] at e_1037_16
  have e_1037_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_17
  simp only [← c64176, ← c64177, k2, ← c64178] at e_1037_17
  have hr := e_1037_17
  simp only [e_1037_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1037 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8817)) := by
  have c64179 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64180 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64181 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64182 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64183 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64184 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1037_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_18
  simp only [← c64179, ← c64180, ← c64181] at e_1037_18
  have e_1037_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_19
  simp only [← c64182, ← c64183, k2, ← c64184] at e_1037_19
  have hr := e_1037_19
  simp only [e_1037_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8818)) := by
  have c64185 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64186 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64187 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64188 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64189 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64190 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_0
  simp only [← c64185, ← c64186, ← c64187] at e_1038_0
  have e_1038_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_1
  simp only [← c64188, ← c64189, k2, ← c64190] at e_1038_1
  have hr := e_1038_1
  simp only [e_1038_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8819)) := by
  have c64191 := (publicBatchWrapper86_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64192 := (publicBatchWrapper86_copies2006 a h).1
  have c64193 := (publicBatchWrapper86_copies2006 a h).2.1
  have c64194 := (publicBatchWrapper86_copies2006 a h).2.2.1
  have c64195 := (publicBatchWrapper86_copies2006 a h).2.2.2.1
  have c64196 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_2
  simp only [← c64191, ← c64192, ← c64193] at e_1038_2
  have e_1038_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_3
  simp only [← c64194, ← c64195, k2, ← c64196] at e_1038_3
  have hr := e_1038_3
  simp only [e_1038_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8820)) := by
  have c64197 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.1
  have c64198 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.1
  have c64199 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.1
  have c64200 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.1
  have c64201 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.1
  have c64202 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_4
  simp only [← c64197, ← c64198, ← c64199] at e_1038_4
  have e_1038_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_5
  simp only [← c64200, ← c64201, k2, ← c64202] at e_1038_5
  have hr := e_1038_5
  simp only [e_1038_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8821)) := by
  have c64203 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64204 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64205 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64206 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64207 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64208 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_6
  simp only [← c64203, ← c64204, ← c64205] at e_1038_6
  have e_1038_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_7
  simp only [← c64206, ← c64207, k2, ← c64208] at e_1038_7
  have hr := e_1038_7
  simp only [e_1038_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8822)) := by
  have c64209 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64210 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64211 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64212 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64213 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64214 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_8
  simp only [← c64209, ← c64210, ← c64211] at e_1038_8
  have e_1038_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_9
  simp only [← c64212, ← c64213, k2, ← c64214] at e_1038_9
  have hr := e_1038_9
  simp only [e_1038_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8823)) := by
  have c64215 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64216 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64217 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64218 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64219 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64220 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_10
  simp only [← c64215, ← c64216, ← c64217] at e_1038_10
  have e_1038_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_11
  simp only [← c64218, ← c64219, k2, ← c64220] at e_1038_11
  have hr := e_1038_11
  simp only [e_1038_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8824)) := by
  have c64221 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64222 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64223 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64224 := (publicBatchWrapper86_copies2007 a h).1
  have c64225 := (publicBatchWrapper86_copies2007 a h).2.1
  have c64226 := (publicBatchWrapper86_copies2007 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_12
  simp only [← c64221, ← c64222, ← c64223] at e_1038_12
  have e_1038_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_13
  simp only [← c64224, ← c64225, k2, ← c64226] at e_1038_13
  have hr := e_1038_13
  simp only [e_1038_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8825)) := by
  have c64227 := (publicBatchWrapper86_copies2007 a h).2.2.2.1
  have c64228 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.1
  have c64229 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.1
  have c64230 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.1
  have c64231 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.1
  have c64232 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_14
  simp only [← c64227, ← c64228, ← c64229] at e_1038_14
  have e_1038_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_15
  simp only [← c64230, ← c64231, k2, ← c64232] at e_1038_15
  have hr := e_1038_15
  simp only [e_1038_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8826)) := by
  have c64233 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.1
  have c64234 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64235 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64236 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64237 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64238 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_16
  simp only [← c64233, ← c64234, ← c64235] at e_1038_16
  have e_1038_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_17
  simp only [← c64236, ← c64237, k2, ← c64238] at e_1038_17
  have hr := e_1038_17
  simp only [e_1038_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8827)) := by
  have c64239 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64240 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64241 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64242 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64243 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64244 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_18
  simp only [← c64239, ← c64240, ← c64241] at e_1038_18
  have e_1038_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_19
  simp only [← c64242, ← c64243, k2, ← c64244] at e_1038_19
  have hr := e_1038_19
  simp only [e_1038_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8828)) := by
  have c64245 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64246 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64247 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64248 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64249 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64250 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_0
  simp only [← c64245, ← c64246, ← c64247] at e_1039_0
  have e_1039_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_1
  simp only [← c64248, ← c64249, k2, ← c64250] at e_1039_1
  have hr := e_1039_1
  simp only [e_1039_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8829)) := by
  have c64251 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64252 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64253 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64254 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64255 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64256 := (publicBatchWrapper86_copies2008 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_2
  simp only [← c64251, ← c64252, ← c64253] at e_1039_2
  have e_1039_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_3
  simp only [← c64254, ← c64255, k2, ← c64256] at e_1039_3
  have hr := e_1039_3
  simp only [e_1039_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8830)) := by
  have c64257 := (publicBatchWrapper86_copies2008 a h).2.1
  have c64258 := (publicBatchWrapper86_copies2008 a h).2.2.1
  have c64259 := (publicBatchWrapper86_copies2008 a h).2.2.2.1
  have c64260 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.1
  have c64261 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.1
  have c64262 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_4
  simp only [← c64257, ← c64258, ← c64259] at e_1039_4
  have e_1039_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_5
  simp only [← c64260, ← c64261, k2, ← c64262] at e_1039_5
  have hr := e_1039_5
  simp only [e_1039_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8831)) := by
  have c64263 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.1
  have c64264 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.1
  have c64265 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.1
  have c64266 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64267 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64268 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_6
  simp only [← c64263, ← c64264, ← c64265] at e_1039_6
  have e_1039_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_7
  simp only [← c64266, ← c64267, k2, ← c64268] at e_1039_7
  have hr := e_1039_7
  simp only [e_1039_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8832)) := by
  have c64269 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64270 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64271 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64272 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64273 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64274 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_8
  simp only [← c64269, ← c64270, ← c64271] at e_1039_8
  have e_1039_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_9
  simp only [← c64272, ← c64273, k2, ← c64274] at e_1039_9
  have hr := e_1039_9
  simp only [e_1039_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8833)) := by
  have c64275 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64276 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64277 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64278 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64279 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64280 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_10
  simp only [← c64275, ← c64276, ← c64277] at e_1039_10
  have e_1039_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_11
  simp only [← c64278, ← c64279, k2, ← c64280] at e_1039_11
  have hr := e_1039_11
  simp only [e_1039_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8834)) := by
  have c64281 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64282 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64283 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64284 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64285 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64286 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_12
  simp only [← c64281, ← c64282, ← c64283] at e_1039_12
  have e_1039_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_13
  simp only [← c64284, ← c64285, k2, ← c64286] at e_1039_13
  have hr := e_1039_13
  simp only [e_1039_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8835)) := by
  have c64287 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64288 := (publicBatchWrapper86_copies2009 a h).1
  have c64289 := (publicBatchWrapper86_copies2009 a h).2.1
  have c64290 := (publicBatchWrapper86_copies2009 a h).2.2.1
  have c64291 := (publicBatchWrapper86_copies2009 a h).2.2.2.1
  have c64292 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_14
  simp only [← c64287, ← c64288, ← c64289] at e_1039_14
  have e_1039_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_15
  simp only [← c64290, ← c64291, k2, ← c64292] at e_1039_15
  have hr := e_1039_15
  simp only [e_1039_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8836)) := by
  have c64293 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.1
  have c64294 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.1
  have c64295 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.1
  have c64296 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.1
  have c64297 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.1
  have c64298 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_16
  simp only [← c64293, ← c64294, ← c64295] at e_1039_16
  have e_1039_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_17
  simp only [← c64296, ← c64297, k2, ← c64298] at e_1039_17
  have hr := e_1039_17
  simp only [e_1039_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8837)) := by
  have c64299 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64300 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64301 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64302 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64303 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64304 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_18
  simp only [← c64299, ← c64300, ← c64301] at e_1039_18
  have e_1039_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_19
  simp only [← c64302, ← c64303, k2, ← c64304] at e_1039_19
  have hr := e_1039_19
  simp only [e_1039_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8838)) := by
  have c64305 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64306 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64307 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64308 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64309 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64310 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_0
  simp only [← c64305, ← c64306, ← c64307] at e_1040_0
  have e_1040_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_1
  simp only [← c64308, ← c64309, k2, ← c64310] at e_1040_1
  have hr := e_1040_1
  simp only [e_1040_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8839)) := by
  have c64311 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64312 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64313 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64314 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64315 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64316 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_2
  simp only [← c64311, ← c64312, ← c64313] at e_1040_2
  have e_1040_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_3
  simp only [← c64314, ← c64315, k2, ← c64316] at e_1040_3
  have hr := e_1040_3
  simp only [e_1040_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8840)) := by
  have c64317 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64318 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64319 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64320 := (publicBatchWrapper86_copies2010 a h).1
  have c64321 := (publicBatchWrapper86_copies2010 a h).2.1
  have c64322 := (publicBatchWrapper86_copies2010 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_4
  simp only [← c64317, ← c64318, ← c64319] at e_1040_4
  have e_1040_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_5
  simp only [← c64320, ← c64321, k2, ← c64322] at e_1040_5
  have hr := e_1040_5
  simp only [e_1040_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8841)) := by
  have c64323 := (publicBatchWrapper86_copies2010 a h).2.2.2.1
  have c64324 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.1
  have c64325 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.1
  have c64326 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.1
  have c64327 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.1
  have c64328 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_6
  simp only [← c64323, ← c64324, ← c64325] at e_1040_6
  have e_1040_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_7
  simp only [← c64326, ← c64327, k2, ← c64328] at e_1040_7
  have hr := e_1040_7
  simp only [e_1040_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8842)) := by
  have c64329 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.1
  have c64330 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64331 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64332 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64333 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64334 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_8
  simp only [← c64329, ← c64330, ← c64331] at e_1040_8
  have e_1040_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_9
  simp only [← c64332, ← c64333, k2, ← c64334] at e_1040_9
  have hr := e_1040_9
  simp only [e_1040_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8843)) := by
  have c64335 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64336 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64337 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64338 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64339 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64340 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_10
  simp only [← c64335, ← c64336, ← c64337] at e_1040_10
  have e_1040_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_11
  simp only [← c64338, ← c64339, k2, ← c64340] at e_1040_11
  have hr := e_1040_11
  simp only [e_1040_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8844)) := by
  have c64341 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64342 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64343 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64344 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64345 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64346 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_12
  simp only [← c64341, ← c64342, ← c64343] at e_1040_12
  have e_1040_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_13
  simp only [← c64344, ← c64345, k2, ← c64346] at e_1040_13
  have hr := e_1040_13
  simp only [e_1040_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8845)) := by
  have c64347 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64348 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64349 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64350 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64351 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64352 := (publicBatchWrapper86_copies2011 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_14
  simp only [← c64347, ← c64348, ← c64349] at e_1040_14
  have e_1040_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_15
  simp only [← c64350, ← c64351, k2, ← c64352] at e_1040_15
  have hr := e_1040_15
  simp only [e_1040_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8846)) := by
  have c64353 := (publicBatchWrapper86_copies2011 a h).2.1
  have c64354 := (publicBatchWrapper86_copies2011 a h).2.2.1
  have c64355 := (publicBatchWrapper86_copies2011 a h).2.2.2.1
  have c64356 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.1
  have c64357 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.1
  have c64358 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_16
  simp only [← c64353, ← c64354, ← c64355] at e_1040_16
  have e_1040_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_17
  simp only [← c64356, ← c64357, k2, ← c64358] at e_1040_17
  have hr := e_1040_17
  simp only [e_1040_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8847)) := by
  have c64359 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.1
  have c64360 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.1
  have c64361 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.1
  have c64362 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64363 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64364 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_18
  simp only [← c64359, ← c64360, ← c64361] at e_1040_18
  have e_1040_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_19
  simp only [← c64362, ← c64363, k2, ← c64364] at e_1040_19
  have hr := e_1040_19
  simp only [e_1040_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8848)) := by
  have c64365 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64366 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64367 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64368 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64369 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64370 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_0
  simp only [← c64365, ← c64366, ← c64367] at e_1041_0
  have e_1041_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_1
  simp only [← c64368, ← c64369, k2, ← c64370] at e_1041_1
  have hr := e_1041_1
  simp only [e_1041_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8849)) := by
  have c64371 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64372 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64373 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64374 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64375 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64376 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_2
  simp only [← c64371, ← c64372, ← c64373] at e_1041_2
  have e_1041_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_3
  simp only [← c64374, ← c64375, k2, ← c64376] at e_1041_3
  have hr := e_1041_3
  simp only [e_1041_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8850)) := by
  have c64377 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64378 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64379 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64380 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64381 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64382 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_4
  simp only [← c64377, ← c64378, ← c64379] at e_1041_4
  have e_1041_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_5
  simp only [← c64380, ← c64381, k2, ← c64382] at e_1041_5
  have hr := e_1041_5
  simp only [e_1041_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8851)) := by
  have c64383 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64384 := (publicBatchWrapper86_copies2012 a h).1
  have c64385 := (publicBatchWrapper86_copies2012 a h).2.1
  have c64386 := (publicBatchWrapper86_copies2012 a h).2.2.1
  have c64387 := (publicBatchWrapper86_copies2012 a h).2.2.2.1
  have c64388 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_6
  simp only [← c64383, ← c64384, ← c64385] at e_1041_6
  have e_1041_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_7
  simp only [← c64386, ← c64387, k2, ← c64388] at e_1041_7
  have hr := e_1041_7
  simp only [e_1041_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8852)) := by
  have c64389 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.1
  have c64390 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.1
  have c64391 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.1
  have c64392 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.1
  have c64393 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.1
  have c64394 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_8
  simp only [← c64389, ← c64390, ← c64391] at e_1041_8
  have e_1041_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_9
  simp only [← c64392, ← c64393, k2, ← c64394] at e_1041_9
  have hr := e_1041_9
  simp only [e_1041_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8853)) := by
  have c64395 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64396 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64397 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64398 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64399 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64400 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_10
  simp only [← c64395, ← c64396, ← c64397] at e_1041_10
  have e_1041_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_11
  simp only [← c64398, ← c64399, k2, ← c64400] at e_1041_11
  have hr := e_1041_11
  simp only [e_1041_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8854)) := by
  have c64401 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64402 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64403 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64404 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64405 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64406 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_12
  simp only [← c64401, ← c64402, ← c64403] at e_1041_12
  have e_1041_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_13
  simp only [← c64404, ← c64405, k2, ← c64406] at e_1041_13
  have hr := e_1041_13
  simp only [e_1041_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8855)) := by
  have c64407 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64408 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64409 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64410 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64411 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64412 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_14
  simp only [← c64407, ← c64408, ← c64409] at e_1041_14
  have e_1041_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_15
  simp only [← c64410, ← c64411, k2, ← c64412] at e_1041_15
  have hr := e_1041_15
  simp only [e_1041_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8856)) := by
  have c64413 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64414 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64415 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64416 := (publicBatchWrapper86_copies2013 a h).1
  have c64417 := (publicBatchWrapper86_copies2013 a h).2.1
  have c64418 := (publicBatchWrapper86_copies2013 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_16
  simp only [← c64413, ← c64414, ← c64415] at e_1041_16
  have e_1041_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_17
  simp only [← c64416, ← c64417, k2, ← c64418] at e_1041_17
  have hr := e_1041_17
  simp only [e_1041_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8857)) := by
  have c64419 := (publicBatchWrapper86_copies2013 a h).2.2.2.1
  have c64420 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.1
  have c64421 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.1
  have c64422 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.1
  have c64423 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.1
  have c64424 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_18
  simp only [← c64419, ← c64420, ← c64421] at e_1041_18
  have e_1041_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_19
  simp only [← c64422, ← c64423, k2, ← c64424] at e_1041_19
  have hr := e_1041_19
  simp only [e_1041_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8858)) := by
  have c64425 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.1
  have c64426 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64427 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64428 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64429 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64430 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_0
  simp only [← c64425, ← c64426, ← c64427] at e_1042_0
  have e_1042_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_1
  simp only [← c64428, ← c64429, k2, ← c64430] at e_1042_1
  have hr := e_1042_1
  simp only [e_1042_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8859)) := by
  have c64431 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64432 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64433 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64434 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64435 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64436 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_2
  simp only [← c64431, ← c64432, ← c64433] at e_1042_2
  have e_1042_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_3
  simp only [← c64434, ← c64435, k2, ← c64436] at e_1042_3
  have hr := e_1042_3
  simp only [e_1042_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8860)) := by
  have c64437 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64438 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64439 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64440 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64441 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64442 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_4
  simp only [← c64437, ← c64438, ← c64439] at e_1042_4
  have e_1042_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_5
  simp only [← c64440, ← c64441, k2, ← c64442] at e_1042_5
  have hr := e_1042_5
  simp only [e_1042_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8861)) := by
  have c64443 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64444 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64445 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64446 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64447 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64448 := (publicBatchWrapper86_copies2014 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_6
  simp only [← c64443, ← c64444, ← c64445] at e_1042_6
  have e_1042_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_7
  simp only [← c64446, ← c64447, k2, ← c64448] at e_1042_7
  have hr := e_1042_7
  simp only [e_1042_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8862)) := by
  have c64449 := (publicBatchWrapper86_copies2014 a h).2.1
  have c64450 := (publicBatchWrapper86_copies2014 a h).2.2.1
  have c64451 := (publicBatchWrapper86_copies2014 a h).2.2.2.1
  have c64452 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.1
  have c64453 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.1
  have c64454 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_8
  simp only [← c64449, ← c64450, ← c64451] at e_1042_8
  have e_1042_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_9
  simp only [← c64452, ← c64453, k2, ← c64454] at e_1042_9
  have hr := e_1042_9
  simp only [e_1042_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8863)) := by
  have c64455 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.1
  have c64456 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.1
  have c64457 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.1
  have c64458 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64459 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64460 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_10
  simp only [← c64455, ← c64456, ← c64457] at e_1042_10
  have e_1042_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_11
  simp only [← c64458, ← c64459, k2, ← c64460] at e_1042_11
  have hr := e_1042_11
  simp only [e_1042_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8864)) := by
  have c64461 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64462 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64463 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64464 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64465 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64466 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_12
  simp only [← c64461, ← c64462, ← c64463] at e_1042_12
  have e_1042_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_13
  simp only [← c64464, ← c64465, k2, ← c64466] at e_1042_13
  have hr := e_1042_13
  simp only [e_1042_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8865)) := by
  have c64467 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64468 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64469 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64470 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64471 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64472 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_14
  simp only [← c64467, ← c64468, ← c64469] at e_1042_14
  have e_1042_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_15
  simp only [← c64470, ← c64471, k2, ← c64472] at e_1042_15
  have hr := e_1042_15
  simp only [e_1042_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8866)) := by
  have c64473 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64474 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64475 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64476 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64477 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64478 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_16
  simp only [← c64473, ← c64474, ← c64475] at e_1042_16
  have e_1042_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_17
  simp only [← c64476, ← c64477, k2, ← c64478] at e_1042_17
  have hr := e_1042_17
  simp only [e_1042_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8867)) := by
  have c64479 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64480 := (publicBatchWrapper86_copies2015 a h).1
  have c64481 := (publicBatchWrapper86_copies2015 a h).2.1
  have c64482 := (publicBatchWrapper86_copies2015 a h).2.2.1
  have c64483 := (publicBatchWrapper86_copies2015 a h).2.2.2.1
  have c64484 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_18
  simp only [← c64479, ← c64480, ← c64481] at e_1042_18
  have e_1042_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_19
  simp only [← c64482, ← c64483, k2, ← c64484] at e_1042_19
  have hr := e_1042_19
  simp only [e_1042_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8868)) := by
  have c64485 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.1
  have c64486 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.1
  have c64487 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.1
  have c64488 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.1
  have c64489 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.1
  have c64490 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_0
  simp only [← c64485, ← c64486, ← c64487] at e_1043_0
  have e_1043_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_1
  simp only [← c64488, ← c64489, k2, ← c64490] at e_1043_1
  have hr := e_1043_1
  simp only [e_1043_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8869)) := by
  have c64491 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64492 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64493 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64494 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64495 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64496 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_2
  simp only [← c64491, ← c64492, ← c64493] at e_1043_2
  have e_1043_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_3
  simp only [← c64494, ← c64495, k2, ← c64496] at e_1043_3
  have hr := e_1043_3
  simp only [e_1043_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8870)) := by
  have c64497 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64498 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64499 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64500 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64501 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64502 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_4
  simp only [← c64497, ← c64498, ← c64499] at e_1043_4
  have e_1043_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_5
  simp only [← c64500, ← c64501, k2, ← c64502] at e_1043_5
  have hr := e_1043_5
  simp only [e_1043_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8871)) := by
  have c64503 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64504 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64505 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64506 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64507 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64508 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_6
  simp only [← c64503, ← c64504, ← c64505] at e_1043_6
  have e_1043_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_7
  simp only [← c64506, ← c64507, k2, ← c64508] at e_1043_7
  have hr := e_1043_7
  simp only [e_1043_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8872)) := by
  have c64509 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64510 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64511 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64512 := (publicBatchWrapper86_copies2016 a h).1
  have c64513 := (publicBatchWrapper86_copies2016 a h).2.1
  have c64514 := (publicBatchWrapper86_copies2016 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_8
  simp only [← c64509, ← c64510, ← c64511] at e_1043_8
  have e_1043_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_9
  simp only [← c64512, ← c64513, k2, ← c64514] at e_1043_9
  have hr := e_1043_9
  simp only [e_1043_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8873)) := by
  have c64515 := (publicBatchWrapper86_copies2016 a h).2.2.2.1
  have c64516 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.1
  have c64517 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.1
  have c64518 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.1
  have c64519 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.1
  have c64520 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_10
  simp only [← c64515, ← c64516, ← c64517] at e_1043_10
  have e_1043_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_11
  simp only [← c64518, ← c64519, k2, ← c64520] at e_1043_11
  have hr := e_1043_11
  simp only [e_1043_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8874)) := by
  have c64521 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.1
  have c64522 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64523 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64524 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64525 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64526 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_12
  simp only [← c64521, ← c64522, ← c64523] at e_1043_12
  have e_1043_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_13
  simp only [← c64524, ← c64525, k2, ← c64526] at e_1043_13
  have hr := e_1043_13
  simp only [e_1043_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8875)) := by
  have c64527 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64528 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64529 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64530 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64531 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64532 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_14
  simp only [← c64527, ← c64528, ← c64529] at e_1043_14
  have e_1043_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_15
  simp only [← c64530, ← c64531, k2, ← c64532] at e_1043_15
  have hr := e_1043_15
  simp only [e_1043_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8876)) := by
  have c64533 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64534 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64535 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64536 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64537 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64538 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_16
  simp only [← c64533, ← c64534, ← c64535] at e_1043_16
  have e_1043_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_17
  simp only [← c64536, ← c64537, k2, ← c64538] at e_1043_17
  have hr := e_1043_17
  simp only [e_1043_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8877)) := by
  have c64539 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64540 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64541 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64542 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64543 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64544 := (publicBatchWrapper86_copies2017 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_18
  simp only [← c64539, ← c64540, ← c64541] at e_1043_18
  have e_1043_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_19
  simp only [← c64542, ← c64543, k2, ← c64544] at e_1043_19
  have hr := e_1043_19
  simp only [e_1043_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8878)) := by
  have c64545 := (publicBatchWrapper86_copies2017 a h).2.1
  have c64546 := (publicBatchWrapper86_copies2017 a h).2.2.1
  have c64547 := (publicBatchWrapper86_copies2017 a h).2.2.2.1
  have c64548 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.1
  have c64549 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.1
  have c64550 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_0
  simp only [← c64545, ← c64546, ← c64547] at e_1044_0
  have e_1044_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_1
  simp only [← c64548, ← c64549, k2, ← c64550] at e_1044_1
  have hr := e_1044_1
  simp only [e_1044_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8879)) := by
  have c64551 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.1
  have c64552 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.1
  have c64553 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.1
  have c64554 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64555 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64556 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_2
  simp only [← c64551, ← c64552, ← c64553] at e_1044_2
  have e_1044_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_3
  simp only [← c64554, ← c64555, k2, ← c64556] at e_1044_3
  have hr := e_1044_3
  simp only [e_1044_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8880)) := by
  have c64557 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64558 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64559 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64560 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64561 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64562 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_4
  simp only [← c64557, ← c64558, ← c64559] at e_1044_4
  have e_1044_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_5
  simp only [← c64560, ← c64561, k2, ← c64562] at e_1044_5
  have hr := e_1044_5
  simp only [e_1044_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8881)) := by
  have c64563 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64564 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64565 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64566 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64567 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64568 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_6
  simp only [← c64563, ← c64564, ← c64565] at e_1044_6
  have e_1044_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_7
  simp only [← c64566, ← c64567, k2, ← c64568] at e_1044_7
  have hr := e_1044_7
  simp only [e_1044_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8882)) := by
  have c64569 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64570 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64571 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64572 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64573 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64574 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_8
  simp only [← c64569, ← c64570, ← c64571] at e_1044_8
  have e_1044_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_9
  simp only [← c64572, ← c64573, k2, ← c64574] at e_1044_9
  have hr := e_1044_9
  simp only [e_1044_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8883)) := by
  have c64575 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64576 := (publicBatchWrapper86_copies2018 a h).1
  have c64577 := (publicBatchWrapper86_copies2018 a h).2.1
  have c64578 := (publicBatchWrapper86_copies2018 a h).2.2.1
  have c64579 := (publicBatchWrapper86_copies2018 a h).2.2.2.1
  have c64580 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_10
  simp only [← c64575, ← c64576, ← c64577] at e_1044_10
  have e_1044_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_11
  simp only [← c64578, ← c64579, k2, ← c64580] at e_1044_11
  have hr := e_1044_11
  simp only [e_1044_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8884)) := by
  have c64581 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.1
  have c64582 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.1
  have c64583 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.1
  have c64584 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.1
  have c64585 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.1
  have c64586 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_12
  simp only [← c64581, ← c64582, ← c64583] at e_1044_12
  have e_1044_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_13
  simp only [← c64584, ← c64585, k2, ← c64586] at e_1044_13
  have hr := e_1044_13
  simp only [e_1044_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8885)) := by
  have c64587 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64588 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64589 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64590 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64591 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64592 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_14
  simp only [← c64587, ← c64588, ← c64589] at e_1044_14
  have e_1044_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_15
  simp only [← c64590, ← c64591, k2, ← c64592] at e_1044_15
  have hr := e_1044_15
  simp only [e_1044_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8886)) := by
  have c64593 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64594 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64595 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64596 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64597 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64598 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_16
  simp only [← c64593, ← c64594, ← c64595] at e_1044_16
  have e_1044_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_17
  simp only [← c64596, ← c64597, k2, ← c64598] at e_1044_17
  have hr := e_1044_17
  simp only [e_1044_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8887)) := by
  have c64599 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64600 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64601 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64602 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64603 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64604 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_18
  simp only [← c64599, ← c64600, ← c64601] at e_1044_18
  have e_1044_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_19
  simp only [← c64602, ← c64603, k2, ← c64604] at e_1044_19
  have hr := e_1044_19
  simp only [e_1044_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8888)) := by
  have c64605 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64606 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64607 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64608 := (publicBatchWrapper86_copies2019 a h).1
  have c64609 := (publicBatchWrapper86_copies2019 a h).2.1
  have c64610 := (publicBatchWrapper86_copies2019 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_0
  simp only [← c64605, ← c64606, ← c64607] at e_1045_0
  have e_1045_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_1
  simp only [← c64608, ← c64609, k2, ← c64610] at e_1045_1
  have hr := e_1045_1
  simp only [e_1045_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8889)) := by
  have c64611 := (publicBatchWrapper86_copies2019 a h).2.2.2.1
  have c64612 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.1
  have c64613 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.1
  have c64614 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.1
  have c64615 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.1
  have c64616 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_2
  simp only [← c64611, ← c64612, ← c64613] at e_1045_2
  have e_1045_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_3
  simp only [← c64614, ← c64615, k2, ← c64616] at e_1045_3
  have hr := e_1045_3
  simp only [e_1045_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8890)) := by
  have c64617 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.1
  have c64618 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64619 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64620 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64621 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64622 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_4
  simp only [← c64617, ← c64618, ← c64619] at e_1045_4
  have e_1045_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_5
  simp only [← c64620, ← c64621, k2, ← c64622] at e_1045_5
  have hr := e_1045_5
  simp only [e_1045_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8891)) := by
  have c64623 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64624 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64625 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64626 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64627 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64628 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_6
  simp only [← c64623, ← c64624, ← c64625] at e_1045_6
  have e_1045_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_7
  simp only [← c64626, ← c64627, k2, ← c64628] at e_1045_7
  have hr := e_1045_7
  simp only [e_1045_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8892)) := by
  have c64629 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64630 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64631 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64632 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64633 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64634 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_8
  simp only [← c64629, ← c64630, ← c64631] at e_1045_8
  have e_1045_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_9
  simp only [← c64632, ← c64633, k2, ← c64634] at e_1045_9
  have hr := e_1045_9
  simp only [e_1045_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8893)) := by
  have c64635 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64636 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64637 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64638 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64639 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64640 := (publicBatchWrapper86_copies2020 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_10
  simp only [← c64635, ← c64636, ← c64637] at e_1045_10
  have e_1045_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_11
  simp only [← c64638, ← c64639, k2, ← c64640] at e_1045_11
  have hr := e_1045_11
  simp only [e_1045_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8894)) := by
  have c64641 := (publicBatchWrapper86_copies2020 a h).2.1
  have c64642 := (publicBatchWrapper86_copies2020 a h).2.2.1
  have c64643 := (publicBatchWrapper86_copies2020 a h).2.2.2.1
  have c64644 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.1
  have c64645 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.1
  have c64646 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_12
  simp only [← c64641, ← c64642, ← c64643] at e_1045_12
  have e_1045_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_13
  simp only [← c64644, ← c64645, k2, ← c64646] at e_1045_13
  have hr := e_1045_13
  simp only [e_1045_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8895)) := by
  have c64647 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.1
  have c64648 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.1
  have c64649 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.1
  have c64650 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64651 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64652 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_14
  simp only [← c64647, ← c64648, ← c64649] at e_1045_14
  have e_1045_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_15
  simp only [← c64650, ← c64651, k2, ← c64652] at e_1045_15
  have hr := e_1045_15
  simp only [e_1045_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8896)) := by
  have c64653 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64654 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64655 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64656 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64657 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64658 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_16
  simp only [← c64653, ← c64654, ← c64655] at e_1045_16
  have e_1045_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_17
  simp only [← c64656, ← c64657, k2, ← c64658] at e_1045_17
  have hr := e_1045_17
  simp only [e_1045_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8897)) := by
  have c64659 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64660 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64661 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64662 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64663 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64664 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_18
  simp only [← c64659, ← c64660, ← c64661] at e_1045_18
  have e_1045_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_19
  simp only [← c64662, ← c64663, k2, ← c64664] at e_1045_19
  have hr := e_1045_19
  simp only [e_1045_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8898)) := by
  have c64665 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64666 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64667 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64668 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64669 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64670 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_0
  simp only [← c64665, ← c64666, ← c64667] at e_1046_0
  have e_1046_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_1
  simp only [← c64668, ← c64669, k2, ← c64670] at e_1046_1
  have hr := e_1046_1
  simp only [e_1046_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8899)) := by
  have c64671 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64672 := (publicBatchWrapper86_copies2021 a h).1
  have c64673 := (publicBatchWrapper86_copies2021 a h).2.1
  have c64674 := (publicBatchWrapper86_copies2021 a h).2.2.1
  have c64675 := (publicBatchWrapper86_copies2021 a h).2.2.2.1
  have c64676 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_2
  simp only [← c64671, ← c64672, ← c64673] at e_1046_2
  have e_1046_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_3
  simp only [← c64674, ← c64675, k2, ← c64676] at e_1046_3
  have hr := e_1046_3
  simp only [e_1046_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8900)) := by
  have c64677 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.1
  have c64678 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.1
  have c64679 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.1
  have c64680 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.1
  have c64681 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.1
  have c64682 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_4
  simp only [← c64677, ← c64678, ← c64679] at e_1046_4
  have e_1046_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_5
  simp only [← c64680, ← c64681, k2, ← c64682] at e_1046_5
  have hr := e_1046_5
  simp only [e_1046_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8901)) := by
  have c64683 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64684 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64685 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64686 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64687 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64688 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_6
  simp only [← c64683, ← c64684, ← c64685] at e_1046_6
  have e_1046_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_7
  simp only [← c64686, ← c64687, k2, ← c64688] at e_1046_7
  have hr := e_1046_7
  simp only [e_1046_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8902)) := by
  have c64689 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64690 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64691 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64692 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64693 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64694 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_8
  simp only [← c64689, ← c64690, ← c64691] at e_1046_8
  have e_1046_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_9
  simp only [← c64692, ← c64693, k2, ← c64694] at e_1046_9
  have hr := e_1046_9
  simp only [e_1046_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8903)) := by
  have c64695 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64696 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64697 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64698 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64699 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64700 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_10
  simp only [← c64695, ← c64696, ← c64697] at e_1046_10
  have e_1046_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_11
  simp only [← c64698, ← c64699, k2, ← c64700] at e_1046_11
  have hr := e_1046_11
  simp only [e_1046_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8904)) := by
  have c64701 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64702 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64703 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64704 := (publicBatchWrapper86_copies2022 a h).1
  have c64705 := (publicBatchWrapper86_copies2022 a h).2.1
  have c64706 := (publicBatchWrapper86_copies2022 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_12
  simp only [← c64701, ← c64702, ← c64703] at e_1046_12
  have e_1046_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_13
  simp only [← c64704, ← c64705, k2, ← c64706] at e_1046_13
  have hr := e_1046_13
  simp only [e_1046_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8905)) := by
  have c64707 := (publicBatchWrapper86_copies2022 a h).2.2.2.1
  have c64708 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.1
  have c64709 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.1
  have c64710 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.1
  have c64711 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.1
  have c64712 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_14
  simp only [← c64707, ← c64708, ← c64709] at e_1046_14
  have e_1046_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_15
  simp only [← c64710, ← c64711, k2, ← c64712] at e_1046_15
  have hr := e_1046_15
  simp only [e_1046_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8906)) := by
  have c64713 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.1
  have c64714 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64715 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64716 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64717 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64718 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_16
  simp only [← c64713, ← c64714, ← c64715] at e_1046_16
  have e_1046_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_17
  simp only [← c64716, ← c64717, k2, ← c64718] at e_1046_17
  have hr := e_1046_17
  simp only [e_1046_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8907)) := by
  have c64719 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64720 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64721 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64722 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64723 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64724 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_18
  simp only [← c64719, ← c64720, ← c64721] at e_1046_18
  have e_1046_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_19
  simp only [← c64722, ← c64723, k2, ← c64724] at e_1046_19
  have hr := e_1046_19
  simp only [e_1046_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8908)) := by
  have c64725 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64726 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64727 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64728 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64729 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64730 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_0
  simp only [← c64725, ← c64726, ← c64727] at e_1047_0
  have e_1047_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_1
  simp only [← c64728, ← c64729, k2, ← c64730] at e_1047_1
  have hr := e_1047_1
  simp only [e_1047_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8909)) := by
  have c64731 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64732 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64733 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64734 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64735 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64736 := (publicBatchWrapper86_copies2023 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_2
  simp only [← c64731, ← c64732, ← c64733] at e_1047_2
  have e_1047_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_3
  simp only [← c64734, ← c64735, k2, ← c64736] at e_1047_3
  have hr := e_1047_3
  simp only [e_1047_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8910)) := by
  have c64737 := (publicBatchWrapper86_copies2023 a h).2.1
  have c64738 := (publicBatchWrapper86_copies2023 a h).2.2.1
  have c64739 := (publicBatchWrapper86_copies2023 a h).2.2.2.1
  have c64740 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.1
  have c64741 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.1
  have c64742 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_4
  simp only [← c64737, ← c64738, ← c64739] at e_1047_4
  have e_1047_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_5
  simp only [← c64740, ← c64741, k2, ← c64742] at e_1047_5
  have hr := e_1047_5
  simp only [e_1047_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8911)) := by
  have c64743 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.1
  have c64744 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.1
  have c64745 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.1
  have c64746 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64747 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64748 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_6
  simp only [← c64743, ← c64744, ← c64745] at e_1047_6
  have e_1047_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_7
  simp only [← c64746, ← c64747, k2, ← c64748] at e_1047_7
  have hr := e_1047_7
  simp only [e_1047_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8912)) := by
  have c64749 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64750 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64751 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64752 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64753 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64754 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_8
  simp only [← c64749, ← c64750, ← c64751] at e_1047_8
  have e_1047_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_9
  simp only [← c64752, ← c64753, k2, ← c64754] at e_1047_9
  have hr := e_1047_9
  simp only [e_1047_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8913)) := by
  have c64755 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64756 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64757 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64758 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64759 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64760 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_10
  simp only [← c64755, ← c64756, ← c64757] at e_1047_10
  have e_1047_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_11
  simp only [← c64758, ← c64759, k2, ← c64760] at e_1047_11
  have hr := e_1047_11
  simp only [e_1047_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8914)) := by
  have c64761 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64762 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64763 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64764 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64765 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64766 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_12
  simp only [← c64761, ← c64762, ← c64763] at e_1047_12
  have e_1047_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_13
  simp only [← c64764, ← c64765, k2, ← c64766] at e_1047_13
  have hr := e_1047_13
  simp only [e_1047_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8915)) := by
  have c64767 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64768 := (publicBatchWrapper86_copies2024 a h).1
  have c64769 := (publicBatchWrapper86_copies2024 a h).2.1
  have c64770 := (publicBatchWrapper86_copies2024 a h).2.2.1
  have c64771 := (publicBatchWrapper86_copies2024 a h).2.2.2.1
  have c64772 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_14
  simp only [← c64767, ← c64768, ← c64769] at e_1047_14
  have e_1047_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_15
  simp only [← c64770, ← c64771, k2, ← c64772] at e_1047_15
  have hr := e_1047_15
  simp only [e_1047_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8916)) := by
  have c64773 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.1
  have c64774 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.1
  have c64775 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.1
  have c64776 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.1
  have c64777 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.1
  have c64778 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_16
  simp only [← c64773, ← c64774, ← c64775] at e_1047_16
  have e_1047_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_17
  simp only [← c64776, ← c64777, k2, ← c64778] at e_1047_17
  have hr := e_1047_17
  simp only [e_1047_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8917)) := by
  have c64779 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64780 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64781 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64782 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64783 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64784 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_18
  simp only [← c64779, ← c64780, ← c64781] at e_1047_18
  have e_1047_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_19
  simp only [← c64782, ← c64783, k2, ← c64784] at e_1047_19
  have hr := e_1047_19
  simp only [e_1047_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8918)) := by
  have c64785 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64786 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64787 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64788 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64789 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64790 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_0
  simp only [← c64785, ← c64786, ← c64787] at e_1048_0
  have e_1048_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_1
  simp only [← c64788, ← c64789, k2, ← c64790] at e_1048_1
  have hr := e_1048_1
  simp only [e_1048_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8919)) := by
  have c64791 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64792 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64793 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64794 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64795 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64796 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_2
  simp only [← c64791, ← c64792, ← c64793] at e_1048_2
  have e_1048_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_3
  simp only [← c64794, ← c64795, k2, ← c64796] at e_1048_3
  have hr := e_1048_3
  simp only [e_1048_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8920)) := by
  have c64797 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64798 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64799 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64800 := (publicBatchWrapper86_copies2025 a h).1
  have c64801 := (publicBatchWrapper86_copies2025 a h).2.1
  have c64802 := (publicBatchWrapper86_copies2025 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_4
  simp only [← c64797, ← c64798, ← c64799] at e_1048_4
  have e_1048_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_5
  simp only [← c64800, ← c64801, k2, ← c64802] at e_1048_5
  have hr := e_1048_5
  simp only [e_1048_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8921)) := by
  have c64803 := (publicBatchWrapper86_copies2025 a h).2.2.2.1
  have c64804 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.1
  have c64805 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.1
  have c64806 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.1
  have c64807 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.1
  have c64808 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_6
  simp only [← c64803, ← c64804, ← c64805] at e_1048_6
  have e_1048_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_7
  simp only [← c64806, ← c64807, k2, ← c64808] at e_1048_7
  have hr := e_1048_7
  simp only [e_1048_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8922)) := by
  have c64809 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.1
  have c64810 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64811 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64812 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64813 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64814 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_8
  simp only [← c64809, ← c64810, ← c64811] at e_1048_8
  have e_1048_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_9
  simp only [← c64812, ← c64813, k2, ← c64814] at e_1048_9
  have hr := e_1048_9
  simp only [e_1048_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8923)) := by
  have c64815 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64816 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64817 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64818 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64819 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64820 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_10
  simp only [← c64815, ← c64816, ← c64817] at e_1048_10
  have e_1048_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_11
  simp only [← c64818, ← c64819, k2, ← c64820] at e_1048_11
  have hr := e_1048_11
  simp only [e_1048_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8924)) := by
  have c64821 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64822 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64823 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64824 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64825 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64826 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_12
  simp only [← c64821, ← c64822, ← c64823] at e_1048_12
  have e_1048_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_13
  simp only [← c64824, ← c64825, k2, ← c64826] at e_1048_13
  have hr := e_1048_13
  simp only [e_1048_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8925)) := by
  have c64827 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64828 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64829 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64830 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64831 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64832 := (publicBatchWrapper86_copies2026 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_14
  simp only [← c64827, ← c64828, ← c64829] at e_1048_14
  have e_1048_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_15
  simp only [← c64830, ← c64831, k2, ← c64832] at e_1048_15
  have hr := e_1048_15
  simp only [e_1048_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8926)) := by
  have c64833 := (publicBatchWrapper86_copies2026 a h).2.1
  have c64834 := (publicBatchWrapper86_copies2026 a h).2.2.1
  have c64835 := (publicBatchWrapper86_copies2026 a h).2.2.2.1
  have c64836 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.1
  have c64837 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.1
  have c64838 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_16
  simp only [← c64833, ← c64834, ← c64835] at e_1048_16
  have e_1048_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_17
  simp only [← c64836, ← c64837, k2, ← c64838] at e_1048_17
  have hr := e_1048_17
  simp only [e_1048_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8927)) := by
  have c64839 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.1
  have c64840 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.1
  have c64841 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.1
  have c64842 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64843 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64844 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_18
  simp only [← c64839, ← c64840, ← c64841] at e_1048_18
  have e_1048_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_19
  simp only [← c64842, ← c64843, k2, ← c64844] at e_1048_19
  have hr := e_1048_19
  simp only [e_1048_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8928)) := by
  have c64845 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64846 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64847 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64848 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64849 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64850 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_0
  simp only [← c64845, ← c64846, ← c64847] at e_1049_0
  have e_1049_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_1
  simp only [← c64848, ← c64849, k2, ← c64850] at e_1049_1
  have hr := e_1049_1
  simp only [e_1049_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8929)) := by
  have c64851 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64852 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64853 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64854 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64855 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64856 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_2
  simp only [← c64851, ← c64852, ← c64853] at e_1049_2
  have e_1049_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_3
  simp only [← c64854, ← c64855, k2, ← c64856] at e_1049_3
  have hr := e_1049_3
  simp only [e_1049_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8930)) := by
  have c64857 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64858 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64859 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64860 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64861 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64862 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_4
  simp only [← c64857, ← c64858, ← c64859] at e_1049_4
  have e_1049_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_5
  simp only [← c64860, ← c64861, k2, ← c64862] at e_1049_5
  have hr := e_1049_5
  simp only [e_1049_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8931)) := by
  have c64863 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64864 := (publicBatchWrapper86_copies2027 a h).1
  have c64865 := (publicBatchWrapper86_copies2027 a h).2.1
  have c64866 := (publicBatchWrapper86_copies2027 a h).2.2.1
  have c64867 := (publicBatchWrapper86_copies2027 a h).2.2.2.1
  have c64868 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_6
  simp only [← c64863, ← c64864, ← c64865] at e_1049_6
  have e_1049_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_7
  simp only [← c64866, ← c64867, k2, ← c64868] at e_1049_7
  have hr := e_1049_7
  simp only [e_1049_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8932)) := by
  have c64869 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.1
  have c64870 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.1
  have c64871 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.1
  have c64872 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.1
  have c64873 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.1
  have c64874 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_8
  simp only [← c64869, ← c64870, ← c64871] at e_1049_8
  have e_1049_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_9
  simp only [← c64872, ← c64873, k2, ← c64874] at e_1049_9
  have hr := e_1049_9
  simp only [e_1049_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8933)) := by
  have c64875 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64876 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64877 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64878 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64879 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64880 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_10
  simp only [← c64875, ← c64876, ← c64877] at e_1049_10
  have e_1049_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_11
  simp only [← c64878, ← c64879, k2, ← c64880] at e_1049_11
  have hr := e_1049_11
  simp only [e_1049_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8934)) := by
  have c64881 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64882 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64883 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64884 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64885 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64886 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_12
  simp only [← c64881, ← c64882, ← c64883] at e_1049_12
  have e_1049_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_13
  simp only [← c64884, ← c64885, k2, ← c64886] at e_1049_13
  have hr := e_1049_13
  simp only [e_1049_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8935)) := by
  have c64887 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64888 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64889 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64890 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64891 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64892 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_14
  simp only [← c64887, ← c64888, ← c64889] at e_1049_14
  have e_1049_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_15
  simp only [← c64890, ← c64891, k2, ← c64892] at e_1049_15
  have hr := e_1049_15
  simp only [e_1049_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8936)) := by
  have c64893 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64894 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64895 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64896 := (publicBatchWrapper86_copies2028 a h).1
  have c64897 := (publicBatchWrapper86_copies2028 a h).2.1
  have c64898 := (publicBatchWrapper86_copies2028 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_16
  simp only [← c64893, ← c64894, ← c64895] at e_1049_16
  have e_1049_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_17
  simp only [← c64896, ← c64897, k2, ← c64898] at e_1049_17
  have hr := e_1049_17
  simp only [e_1049_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8937)) := by
  have c64899 := (publicBatchWrapper86_copies2028 a h).2.2.2.1
  have c64900 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.1
  have c64901 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.1
  have c64902 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.1
  have c64903 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.1
  have c64904 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_18
  simp only [← c64899, ← c64900, ← c64901] at e_1049_18
  have e_1049_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_19
  simp only [← c64902, ← c64903, k2, ← c64904] at e_1049_19
  have hr := e_1049_19
  simp only [e_1049_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8938)) := by
  have c64905 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.1
  have c64906 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64907 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64908 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64909 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64910 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_0
  simp only [← c64905, ← c64906, ← c64907] at e_1050_0
  have e_1050_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_1
  simp only [← c64908, ← c64909, k2, ← c64910] at e_1050_1
  have hr := e_1050_1
  simp only [e_1050_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8939)) := by
  have c64911 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64912 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64913 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64914 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64915 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64916 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_2
  simp only [← c64911, ← c64912, ← c64913] at e_1050_2
  have e_1050_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_3
  simp only [← c64914, ← c64915, k2, ← c64916] at e_1050_3
  have hr := e_1050_3
  simp only [e_1050_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8940)) := by
  have c64917 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64918 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64919 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64920 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64921 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64922 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_4
  simp only [← c64917, ← c64918, ← c64919] at e_1050_4
  have e_1050_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_5
  simp only [← c64920, ← c64921, k2, ← c64922] at e_1050_5
  have hr := e_1050_5
  simp only [e_1050_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8941)) := by
  have c64923 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64924 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64925 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64926 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64927 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64928 := (publicBatchWrapper86_copies2029 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_6
  simp only [← c64923, ← c64924, ← c64925] at e_1050_6
  have e_1050_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_7
  simp only [← c64926, ← c64927, k2, ← c64928] at e_1050_7
  have hr := e_1050_7
  simp only [e_1050_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8942)) := by
  have c64929 := (publicBatchWrapper86_copies2029 a h).2.1
  have c64930 := (publicBatchWrapper86_copies2029 a h).2.2.1
  have c64931 := (publicBatchWrapper86_copies2029 a h).2.2.2.1
  have c64932 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.1
  have c64933 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.1
  have c64934 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_8
  simp only [← c64929, ← c64930, ← c64931] at e_1050_8
  have e_1050_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_9
  simp only [← c64932, ← c64933, k2, ← c64934] at e_1050_9
  have hr := e_1050_9
  simp only [e_1050_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8943)) := by
  have c64935 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.1
  have c64936 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.1
  have c64937 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.1
  have c64938 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64939 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64940 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_10
  simp only [← c64935, ← c64936, ← c64937] at e_1050_10
  have e_1050_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_11
  simp only [← c64938, ← c64939, k2, ← c64940] at e_1050_11
  have hr := e_1050_11
  simp only [e_1050_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8944)) := by
  have c64941 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64942 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64943 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64944 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64945 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64946 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_12
  simp only [← c64941, ← c64942, ← c64943] at e_1050_12
  have e_1050_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_13
  simp only [← c64944, ← c64945, k2, ← c64946] at e_1050_13
  have hr := e_1050_13
  simp only [e_1050_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8945)) := by
  have c64947 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64948 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64949 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64950 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64951 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64952 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_14
  simp only [← c64947, ← c64948, ← c64949] at e_1050_14
  have e_1050_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_15
  simp only [← c64950, ← c64951, k2, ← c64952] at e_1050_15
  have hr := e_1050_15
  simp only [e_1050_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8946)) := by
  have c64953 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64954 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64955 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64956 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64957 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64958 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_16
  simp only [← c64953, ← c64954, ← c64955] at e_1050_16
  have e_1050_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_17
  simp only [← c64956, ← c64957, k2, ← c64958] at e_1050_17
  have hr := e_1050_17
  simp only [e_1050_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8947)) := by
  have c64959 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64960 := (publicBatchWrapper86_copies2030 a h).1
  have c64961 := (publicBatchWrapper86_copies2030 a h).2.1
  have c64962 := (publicBatchWrapper86_copies2030 a h).2.2.1
  have c64963 := (publicBatchWrapper86_copies2030 a h).2.2.2.1
  have c64964 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_18
  simp only [← c64959, ← c64960, ← c64961] at e_1050_18
  have e_1050_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_19
  simp only [← c64962, ← c64963, k2, ← c64964] at e_1050_19
  have hr := e_1050_19
  simp only [e_1050_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8948)) := by
  have c64965 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.1
  have c64966 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.1
  have c64967 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.1
  have c64968 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.1
  have c64969 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.1
  have c64970 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_0
  simp only [← c64965, ← c64966, ← c64967] at e_1051_0
  have e_1051_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_1
  simp only [← c64968, ← c64969, k2, ← c64970] at e_1051_1
  have hr := e_1051_1
  simp only [e_1051_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8949)) := by
  have c64971 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64972 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64973 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64974 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64975 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64976 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_2
  simp only [← c64971, ← c64972, ← c64973] at e_1051_2
  have e_1051_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_3
  simp only [← c64974, ← c64975, k2, ← c64976] at e_1051_3
  have hr := e_1051_3
  simp only [e_1051_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8950)) := by
  have c64977 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64978 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64979 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64980 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64981 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64982 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_4
  simp only [← c64977, ← c64978, ← c64979] at e_1051_4
  have e_1051_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_5
  simp only [← c64980, ← c64981, k2, ← c64982] at e_1051_5
  have hr := e_1051_5
  simp only [e_1051_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8951)) := by
  have c64983 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64984 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64985 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64986 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64987 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64988 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_6
  simp only [← c64983, ← c64984, ← c64985] at e_1051_6
  have e_1051_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_7
  simp only [← c64986, ← c64987, k2, ← c64988] at e_1051_7
  have hr := e_1051_7
  simp only [e_1051_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8952)) := by
  have c64989 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64990 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64991 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64992 := (publicBatchWrapper86_copies2031 a h).1
  have c64993 := (publicBatchWrapper86_copies2031 a h).2.1
  have c64994 := (publicBatchWrapper86_copies2031 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_8
  simp only [← c64989, ← c64990, ← c64991] at e_1051_8
  have e_1051_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_9
  simp only [← c64992, ← c64993, k2, ← c64994] at e_1051_9
  have hr := e_1051_9
  simp only [e_1051_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8953)) := by
  have c64995 := (publicBatchWrapper86_copies2031 a h).2.2.2.1
  have c64996 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.1
  have c64997 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.1
  have c64998 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.1
  have c64999 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.1
  have c65000 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_10
  simp only [← c64995, ← c64996, ← c64997] at e_1051_10
  have e_1051_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_11
  simp only [← c64998, ← c64999, k2, ← c65000] at e_1051_11
  have hr := e_1051_11
  simp only [e_1051_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8954)) := by
  have c65001 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.1
  have c65002 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65003 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65004 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65005 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65006 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_12
  simp only [← c65001, ← c65002, ← c65003] at e_1051_12
  have e_1051_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_13
  simp only [← c65004, ← c65005, k2, ← c65006] at e_1051_13
  have hr := e_1051_13
  simp only [e_1051_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8955)) := by
  have c65007 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65008 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65009 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65010 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65011 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65012 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_14
  simp only [← c65007, ← c65008, ← c65009] at e_1051_14
  have e_1051_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_15
  simp only [← c65010, ← c65011, k2, ← c65012] at e_1051_15
  have hr := e_1051_15
  simp only [e_1051_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8956)) := by
  have c65013 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65014 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65015 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65016 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65017 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65018 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_16
  simp only [← c65013, ← c65014, ← c65015] at e_1051_16
  have e_1051_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_17
  simp only [← c65016, ← c65017, k2, ← c65018] at e_1051_17
  have hr := e_1051_17
  simp only [e_1051_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8957)) := by
  have c65019 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65020 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65021 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65022 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65023 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65024 := (publicBatchWrapper86_copies2032 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_18
  simp only [← c65019, ← c65020, ← c65021] at e_1051_18
  have e_1051_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_19
  simp only [← c65022, ← c65023, k2, ← c65024] at e_1051_19
  have hr := e_1051_19
  simp only [e_1051_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8958)) := by
  have c65025 := (publicBatchWrapper86_copies2032 a h).2.1
  have c65026 := (publicBatchWrapper86_copies2032 a h).2.2.1
  have c65027 := (publicBatchWrapper86_copies2032 a h).2.2.2.1
  have c65028 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.1
  have c65029 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.1
  have c65030 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_0
  simp only [← c65025, ← c65026, ← c65027] at e_1052_0
  have e_1052_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_1
  simp only [← c65028, ← c65029, k2, ← c65030] at e_1052_1
  have hr := e_1052_1
  simp only [e_1052_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8959)) := by
  have c65031 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.1
  have c65032 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.1
  have c65033 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.1
  have c65034 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65035 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65036 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_2
  simp only [← c65031, ← c65032, ← c65033] at e_1052_2
  have e_1052_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_3
  simp only [← c65034, ← c65035, k2, ← c65036] at e_1052_3
  have hr := e_1052_3
  simp only [e_1052_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8960)) := by
  have c65037 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65038 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65039 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65040 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65041 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65042 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_4
  simp only [← c65037, ← c65038, ← c65039] at e_1052_4
  have e_1052_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_5
  simp only [← c65040, ← c65041, k2, ← c65042] at e_1052_5
  have hr := e_1052_5
  simp only [e_1052_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8961)) := by
  have c65043 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65044 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65045 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65046 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65047 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65048 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_6
  simp only [← c65043, ← c65044, ← c65045] at e_1052_6
  have e_1052_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_7
  simp only [← c65046, ← c65047, k2, ← c65048] at e_1052_7
  have hr := e_1052_7
  simp only [e_1052_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8962)) := by
  have c65049 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65050 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65051 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65052 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65053 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65054 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_8
  simp only [← c65049, ← c65050, ← c65051] at e_1052_8
  have e_1052_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_9
  simp only [← c65052, ← c65053, k2, ← c65054] at e_1052_9
  have hr := e_1052_9
  simp only [e_1052_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8963)) := by
  have c65055 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65056 := (publicBatchWrapper86_copies2033 a h).1
  have c65057 := (publicBatchWrapper86_copies2033 a h).2.1
  have c65058 := (publicBatchWrapper86_copies2033 a h).2.2.1
  have c65059 := (publicBatchWrapper86_copies2033 a h).2.2.2.1
  have c65060 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_10
  simp only [← c65055, ← c65056, ← c65057] at e_1052_10
  have e_1052_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_11
  simp only [← c65058, ← c65059, k2, ← c65060] at e_1052_11
  have hr := e_1052_11
  simp only [e_1052_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8964)) := by
  have c65061 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.1
  have c65062 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.1
  have c65063 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.1
  have c65064 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.1
  have c65065 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.1
  have c65066 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_12
  simp only [← c65061, ← c65062, ← c65063] at e_1052_12
  have e_1052_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_13
  simp only [← c65064, ← c65065, k2, ← c65066] at e_1052_13
  have hr := e_1052_13
  simp only [e_1052_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8965)) := by
  have c65067 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65068 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65069 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65070 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65071 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65072 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_14
  simp only [← c65067, ← c65068, ← c65069] at e_1052_14
  have e_1052_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_15
  simp only [← c65070, ← c65071, k2, ← c65072] at e_1052_15
  have hr := e_1052_15
  simp only [e_1052_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8966)) := by
  have c65073 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65074 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65075 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65076 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65077 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65078 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_16
  simp only [← c65073, ← c65074, ← c65075] at e_1052_16
  have e_1052_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_17
  simp only [← c65076, ← c65077, k2, ← c65078] at e_1052_17
  have hr := e_1052_17
  simp only [e_1052_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8967)) := by
  have c65079 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65080 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65081 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65082 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65083 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65084 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_18
  simp only [← c65079, ← c65080, ← c65081] at e_1052_18
  have e_1052_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_19
  simp only [← c65082, ← c65083, k2, ← c65084] at e_1052_19
  have hr := e_1052_19
  simp only [e_1052_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8968)) := by
  have c65085 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65086 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65087 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65088 := (publicBatchWrapper86_copies2034 a h).1
  have c65089 := (publicBatchWrapper86_copies2034 a h).2.1
  have c65090 := (publicBatchWrapper86_copies2034 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_0
  simp only [← c65085, ← c65086, ← c65087] at e_1053_0
  have e_1053_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_1
  simp only [← c65088, ← c65089, k2, ← c65090] at e_1053_1
  have hr := e_1053_1
  simp only [e_1053_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8969)) := by
  have c65091 := (publicBatchWrapper86_copies2034 a h).2.2.2.1
  have c65092 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.1
  have c65093 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.1
  have c65094 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.1
  have c65095 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.1
  have c65096 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_2
  simp only [← c65091, ← c65092, ← c65093] at e_1053_2
  have e_1053_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_3
  simp only [← c65094, ← c65095, k2, ← c65096] at e_1053_3
  have hr := e_1053_3
  simp only [e_1053_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9010)) := by
  have c65097 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.1
  have c65098 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65099 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65100 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65101 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65102 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_4
  simp only [← c65097, ← c65098, ← c65099] at e_1053_4
  have e_1053_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_5
  simp only [← c65100, ← c65101, k2, ← c65102] at e_1053_5
  have hr := e_1053_5
  simp only [e_1053_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9011)) := by
  have c65103 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65104 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65105 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65106 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65107 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65108 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_6
  simp only [← c65103, ← c65104, ← c65105] at e_1053_6
  have e_1053_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_7
  simp only [← c65106, ← c65107, k2, ← c65108] at e_1053_7
  have hr := e_1053_7
  simp only [e_1053_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9012)) := by
  have c65109 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65110 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65111 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65112 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65113 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65114 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_8
  simp only [← c65109, ← c65110, ← c65111] at e_1053_8
  have e_1053_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_9
  simp only [← c65112, ← c65113, k2, ← c65114] at e_1053_9
  have hr := e_1053_9
  simp only [e_1053_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9013)) := by
  have c65115 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65116 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65117 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65118 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65119 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65120 := (publicBatchWrapper86_copies2035 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_10
  simp only [← c65115, ← c65116, ← c65117] at e_1053_10
  have e_1053_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_11
  simp only [← c65118, ← c65119, k2, ← c65120] at e_1053_11
  have hr := e_1053_11
  simp only [e_1053_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9014)) := by
  have c65121 := (publicBatchWrapper86_copies2035 a h).2.1
  have c65122 := (publicBatchWrapper86_copies2035 a h).2.2.1
  have c65123 := (publicBatchWrapper86_copies2035 a h).2.2.2.1
  have c65124 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.1
  have c65125 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.1
  have c65126 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_12
  simp only [← c65121, ← c65122, ← c65123] at e_1053_12
  have e_1053_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_13
  simp only [← c65124, ← c65125, k2, ← c65126] at e_1053_13
  have hr := e_1053_13
  simp only [e_1053_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9015)) := by
  have c65127 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.1
  have c65128 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.1
  have c65129 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.1
  have c65130 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65131 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65132 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_14
  simp only [← c65127, ← c65128, ← c65129] at e_1053_14
  have e_1053_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_15
  simp only [← c65130, ← c65131, k2, ← c65132] at e_1053_15
  have hr := e_1053_15
  simp only [e_1053_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9016)) := by
  have c65133 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65134 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65135 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65136 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65137 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65138 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_16
  simp only [← c65133, ← c65134, ← c65135] at e_1053_16
  have e_1053_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_17
  simp only [← c65136, ← c65137, k2, ← c65138] at e_1053_17
  have hr := e_1053_17
  simp only [e_1053_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9017)) := by
  have c65139 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65140 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65141 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65142 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65143 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65144 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_18
  simp only [← c65139, ← c65140, ← c65141] at e_1053_18
  have e_1053_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_19
  simp only [← c65142, ← c65143, k2, ← c65144] at e_1053_19
  have hr := e_1053_19
  simp only [e_1053_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9018)) := by
  have c65145 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65146 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65147 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65148 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65149 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65150 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_0
  simp only [← c65145, ← c65146, ← c65147] at e_1054_0
  have e_1054_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_1
  simp only [← c65148, ← c65149, k2, ← c65150] at e_1054_1
  have hr := e_1054_1
  simp only [e_1054_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9019)) := by
  have c65151 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65152 := (publicBatchWrapper86_copies2036 a h).1
  have c65153 := (publicBatchWrapper86_copies2036 a h).2.1
  have c65154 := (publicBatchWrapper86_copies2036 a h).2.2.1
  have c65155 := (publicBatchWrapper86_copies2036 a h).2.2.2.1
  have c65156 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_2
  simp only [← c65151, ← c65152, ← c65153] at e_1054_2
  have e_1054_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_3
  simp only [← c65154, ← c65155, k2, ← c65156] at e_1054_3
  have hr := e_1054_3
  simp only [e_1054_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9020)) := by
  have c65157 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.1
  have c65158 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.1
  have c65159 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.1
  have c65160 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.1
  have c65161 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.1
  have c65162 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_4
  simp only [← c65157, ← c65158, ← c65159] at e_1054_4
  have e_1054_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_5
  simp only [← c65160, ← c65161, k2, ← c65162] at e_1054_5
  have hr := e_1054_5
  simp only [e_1054_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9021)) := by
  have c65163 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65164 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65165 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65166 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65167 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65168 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_6
  simp only [← c65163, ← c65164, ← c65165] at e_1054_6
  have e_1054_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_7
  simp only [← c65166, ← c65167, k2, ← c65168] at e_1054_7
  have hr := e_1054_7
  simp only [e_1054_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9022)) := by
  have c65169 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65170 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65171 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65172 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65173 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65174 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_8
  simp only [← c65169, ← c65170, ← c65171] at e_1054_8
  have e_1054_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_9
  simp only [← c65172, ← c65173, k2, ← c65174] at e_1054_9
  have hr := e_1054_9
  simp only [e_1054_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9023)) := by
  have c65175 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65176 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65177 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65178 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65179 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65180 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_10
  simp only [← c65175, ← c65176, ← c65177] at e_1054_10
  have e_1054_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_11
  simp only [← c65178, ← c65179, k2, ← c65180] at e_1054_11
  have hr := e_1054_11
  simp only [e_1054_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9024)) := by
  have c65181 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65182 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65183 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65184 := (publicBatchWrapper86_copies2037 a h).1
  have c65185 := (publicBatchWrapper86_copies2037 a h).2.1
  have c65186 := (publicBatchWrapper86_copies2037 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_12
  simp only [← c65181, ← c65182, ← c65183] at e_1054_12
  have e_1054_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_13
  simp only [← c65184, ← c65185, k2, ← c65186] at e_1054_13
  have hr := e_1054_13
  simp only [e_1054_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9025)) := by
  have c65187 := (publicBatchWrapper86_copies2037 a h).2.2.2.1
  have c65188 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.1
  have c65189 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.1
  have c65190 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.1
  have c65191 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.1
  have c65192 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_14
  simp only [← c65187, ← c65188, ← c65189] at e_1054_14
  have e_1054_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_15
  simp only [← c65190, ← c65191, k2, ← c65192] at e_1054_15
  have hr := e_1054_15
  simp only [e_1054_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9026)) := by
  have c65193 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.1
  have c65194 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65195 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65196 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65197 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65198 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_16
  simp only [← c65193, ← c65194, ← c65195] at e_1054_16
  have e_1054_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_17
  simp only [← c65196, ← c65197, k2, ← c65198] at e_1054_17
  have hr := e_1054_17
  simp only [e_1054_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9027)) := by
  have c65199 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65200 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65201 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65202 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65203 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65204 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_18
  simp only [← c65199, ← c65200, ← c65201] at e_1054_18
  have e_1054_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_19
  simp only [← c65202, ← c65203, k2, ← c65204] at e_1054_19
  have hr := e_1054_19
  simp only [e_1054_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9028)) := by
  have c65205 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65206 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65207 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65208 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65209 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65210 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_0
  simp only [← c65205, ← c65206, ← c65207] at e_1055_0
  have e_1055_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_1
  simp only [← c65208, ← c65209, k2, ← c65210] at e_1055_1
  have hr := e_1055_1
  simp only [e_1055_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9029)) := by
  have c65211 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65212 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65213 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65214 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65215 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65216 := (publicBatchWrapper86_copies2038 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_2
  simp only [← c65211, ← c65212, ← c65213] at e_1055_2
  have e_1055_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_3
  simp only [← c65214, ← c65215, k2, ← c65216] at e_1055_3
  have hr := e_1055_3
  simp only [e_1055_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9030)) := by
  have c65217 := (publicBatchWrapper86_copies2038 a h).2.1
  have c65218 := (publicBatchWrapper86_copies2038 a h).2.2.1
  have c65219 := (publicBatchWrapper86_copies2038 a h).2.2.2.1
  have c65220 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.1
  have c65221 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.1
  have c65222 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_4
  simp only [← c65217, ← c65218, ← c65219] at e_1055_4
  have e_1055_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_5
  simp only [← c65220, ← c65221, k2, ← c65222] at e_1055_5
  have hr := e_1055_5
  simp only [e_1055_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9031)) := by
  have c65223 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.1
  have c65224 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.1
  have c65225 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.1
  have c65226 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65227 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65228 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_6
  simp only [← c65223, ← c65224, ← c65225] at e_1055_6
  have e_1055_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_7
  simp only [← c65226, ← c65227, k2, ← c65228] at e_1055_7
  have hr := e_1055_7
  simp only [e_1055_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9032)) := by
  have c65229 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65230 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65231 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65232 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65233 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65234 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_8
  simp only [← c65229, ← c65230, ← c65231] at e_1055_8
  have e_1055_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_9
  simp only [← c65232, ← c65233, k2, ← c65234] at e_1055_9
  have hr := e_1055_9
  simp only [e_1055_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9033)) := by
  have c65235 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65236 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65237 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65238 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65239 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65240 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_10
  simp only [← c65235, ← c65236, ← c65237] at e_1055_10
  have e_1055_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_11
  simp only [← c65238, ← c65239, k2, ← c65240] at e_1055_11
  have hr := e_1055_11
  simp only [e_1055_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9034)) := by
  have c65241 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65242 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65243 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65244 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65245 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65246 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_12
  simp only [← c65241, ← c65242, ← c65243] at e_1055_12
  have e_1055_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_13
  simp only [← c65244, ← c65245, k2, ← c65246] at e_1055_13
  have hr := e_1055_13
  simp only [e_1055_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9035)) := by
  have c65247 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65248 := (publicBatchWrapper86_copies2039 a h).1
  have c65249 := (publicBatchWrapper86_copies2039 a h).2.1
  have c65250 := (publicBatchWrapper86_copies2039 a h).2.2.1
  have c65251 := (publicBatchWrapper86_copies2039 a h).2.2.2.1
  have c65252 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_14
  simp only [← c65247, ← c65248, ← c65249] at e_1055_14
  have e_1055_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_15
  simp only [← c65250, ← c65251, k2, ← c65252] at e_1055_15
  have hr := e_1055_15
  simp only [e_1055_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9036)) := by
  have c65253 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.1
  have c65254 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.1
  have c65255 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.1
  have c65256 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.1
  have c65257 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.1
  have c65258 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_16
  simp only [← c65253, ← c65254, ← c65255] at e_1055_16
  have e_1055_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_17
  simp only [← c65256, ← c65257, k2, ← c65258] at e_1055_17
  have hr := e_1055_17
  simp only [e_1055_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9037)) := by
  have c65259 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65260 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65261 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65262 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65263 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65264 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_18
  simp only [← c65259, ← c65260, ← c65261] at e_1055_18
  have e_1055_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_19
  simp only [← c65262, ← c65263, k2, ← c65264] at e_1055_19
  have hr := e_1055_19
  simp only [e_1055_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9038)) := by
  have c65265 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65266 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65267 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65268 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65269 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65270 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_0
  simp only [← c65265, ← c65266, ← c65267] at e_1056_0
  have e_1056_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_1
  simp only [← c65268, ← c65269, k2, ← c65270] at e_1056_1
  have hr := e_1056_1
  simp only [e_1056_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9039)) := by
  have c65271 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65272 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65273 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65274 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65275 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65276 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_2
  simp only [← c65271, ← c65272, ← c65273] at e_1056_2
  have e_1056_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_3
  simp only [← c65274, ← c65275, k2, ← c65276] at e_1056_3
  have hr := e_1056_3
  simp only [e_1056_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9040)) := by
  have c65277 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65278 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65279 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65280 := (publicBatchWrapper86_copies2040 a h).1
  have c65281 := (publicBatchWrapper86_copies2040 a h).2.1
  have c65282 := (publicBatchWrapper86_copies2040 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_4
  simp only [← c65277, ← c65278, ← c65279] at e_1056_4
  have e_1056_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_5
  simp only [← c65280, ← c65281, k2, ← c65282] at e_1056_5
  have hr := e_1056_5
  simp only [e_1056_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9041)) := by
  have c65283 := (publicBatchWrapper86_copies2040 a h).2.2.2.1
  have c65284 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.1
  have c65285 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.1
  have c65286 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.1
  have c65287 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.1
  have c65288 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_6
  simp only [← c65283, ← c65284, ← c65285] at e_1056_6
  have e_1056_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_7
  simp only [← c65286, ← c65287, k2, ← c65288] at e_1056_7
  have hr := e_1056_7
  simp only [e_1056_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9042)) := by
  have c65289 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.1
  have c65290 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65291 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65292 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65293 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65294 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_8
  simp only [← c65289, ← c65290, ← c65291] at e_1056_8
  have e_1056_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_9
  simp only [← c65292, ← c65293, k2, ← c65294] at e_1056_9
  have hr := e_1056_9
  simp only [e_1056_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9043)) := by
  have c65295 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65296 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65297 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65298 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65299 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65300 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_10
  simp only [← c65295, ← c65296, ← c65297] at e_1056_10
  have e_1056_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_11
  simp only [← c65298, ← c65299, k2, ← c65300] at e_1056_11
  have hr := e_1056_11
  simp only [e_1056_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9044)) := by
  have c65301 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65302 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65303 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65304 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65305 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65306 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_12
  simp only [← c65301, ← c65302, ← c65303] at e_1056_12
  have e_1056_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_13
  simp only [← c65304, ← c65305, k2, ← c65306] at e_1056_13
  have hr := e_1056_13
  simp only [e_1056_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9045)) := by
  have c65307 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65308 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65309 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65310 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65311 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65312 := (publicBatchWrapper86_copies2041 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_14
  simp only [← c65307, ← c65308, ← c65309] at e_1056_14
  have e_1056_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_15
  simp only [← c65310, ← c65311, k2, ← c65312] at e_1056_15
  have hr := e_1056_15
  simp only [e_1056_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9046)) := by
  have c65313 := (publicBatchWrapper86_copies2041 a h).2.1
  have c65314 := (publicBatchWrapper86_copies2041 a h).2.2.1
  have c65315 := (publicBatchWrapper86_copies2041 a h).2.2.2.1
  have c65316 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.1
  have c65317 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.1
  have c65318 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_16
  simp only [← c65313, ← c65314, ← c65315] at e_1056_16
  have e_1056_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_17
  simp only [← c65316, ← c65317, k2, ← c65318] at e_1056_17
  have hr := e_1056_17
  simp only [e_1056_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9047)) := by
  have c65319 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.1
  have c65320 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.1
  have c65321 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.1
  have c65322 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65323 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65324 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_18
  simp only [← c65319, ← c65320, ← c65321] at e_1056_18
  have e_1056_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_19
  simp only [← c65322, ← c65323, k2, ← c65324] at e_1056_19
  have hr := e_1056_19
  simp only [e_1056_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9048)) := by
  have c65325 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65326 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65327 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65328 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65329 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65330 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_0
  simp only [← c65325, ← c65326, ← c65327] at e_1057_0
  have e_1057_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_1
  simp only [← c65328, ← c65329, k2, ← c65330] at e_1057_1
  have hr := e_1057_1
  simp only [e_1057_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9049)) := by
  have c65331 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65332 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65333 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65334 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65335 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65336 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_2
  simp only [← c65331, ← c65332, ← c65333] at e_1057_2
  have e_1057_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_3
  simp only [← c65334, ← c65335, k2, ← c65336] at e_1057_3
  have hr := e_1057_3
  simp only [e_1057_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9050)) := by
  have c65337 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65338 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65339 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65340 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65341 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65342 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_4
  simp only [← c65337, ← c65338, ← c65339] at e_1057_4
  have e_1057_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_5
  simp only [← c65340, ← c65341, k2, ← c65342] at e_1057_5
  have hr := e_1057_5
  simp only [e_1057_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9051)) := by
  have c65343 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65344 := (publicBatchWrapper86_copies2042 a h).1
  have c65345 := (publicBatchWrapper86_copies2042 a h).2.1
  have c65346 := (publicBatchWrapper86_copies2042 a h).2.2.1
  have c65347 := (publicBatchWrapper86_copies2042 a h).2.2.2.1
  have c65348 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_6
  simp only [← c65343, ← c65344, ← c65345] at e_1057_6
  have e_1057_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_7
  simp only [← c65346, ← c65347, k2, ← c65348] at e_1057_7
  have hr := e_1057_7
  simp only [e_1057_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9052)) := by
  have c65349 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.1
  have c65350 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.1
  have c65351 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.1
  have c65352 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.1
  have c65353 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.1
  have c65354 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_8
  simp only [← c65349, ← c65350, ← c65351] at e_1057_8
  have e_1057_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_9
  simp only [← c65352, ← c65353, k2, ← c65354] at e_1057_9
  have hr := e_1057_9
  simp only [e_1057_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9053)) := by
  have c65355 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65356 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65357 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65358 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65359 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65360 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_10
  simp only [← c65355, ← c65356, ← c65357] at e_1057_10
  have e_1057_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_11
  simp only [← c65358, ← c65359, k2, ← c65360] at e_1057_11
  have hr := e_1057_11
  simp only [e_1057_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9054)) := by
  have c65361 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65362 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65363 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65364 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65365 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65366 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_12
  simp only [← c65361, ← c65362, ← c65363] at e_1057_12
  have e_1057_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_13
  simp only [← c65364, ← c65365, k2, ← c65366] at e_1057_13
  have hr := e_1057_13
  simp only [e_1057_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9055)) := by
  have c65367 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65368 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65369 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65370 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65371 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65372 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_14
  simp only [← c65367, ← c65368, ← c65369] at e_1057_14
  have e_1057_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_15
  simp only [← c65370, ← c65371, k2, ← c65372] at e_1057_15
  have hr := e_1057_15
  simp only [e_1057_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9056)) := by
  have c65373 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65374 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65375 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65376 := (publicBatchWrapper86_copies2043 a h).1
  have c65377 := (publicBatchWrapper86_copies2043 a h).2.1
  have c65378 := (publicBatchWrapper86_copies2043 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_16
  simp only [← c65373, ← c65374, ← c65375] at e_1057_16
  have e_1057_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_17
  simp only [← c65376, ← c65377, k2, ← c65378] at e_1057_17
  have hr := e_1057_17
  simp only [e_1057_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9057)) := by
  have c65379 := (publicBatchWrapper86_copies2043 a h).2.2.2.1
  have c65380 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.1
  have c65381 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.1
  have c65382 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.1
  have c65383 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.1
  have c65384 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_18
  simp only [← c65379, ← c65380, ← c65381] at e_1057_18
  have e_1057_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_19
  simp only [← c65382, ← c65383, k2, ← c65384] at e_1057_19
  have hr := e_1057_19
  simp only [e_1057_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9058)) := by
  have c65385 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.1
  have c65386 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65387 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65388 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65389 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65390 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_0
  simp only [← c65385, ← c65386, ← c65387] at e_1058_0
  have e_1058_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_1
  simp only [← c65388, ← c65389, k2, ← c65390] at e_1058_1
  have hr := e_1058_1
  simp only [e_1058_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9059)) := by
  have c65391 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65392 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65393 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65394 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65395 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65396 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_2
  simp only [← c65391, ← c65392, ← c65393] at e_1058_2
  have e_1058_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_3
  simp only [← c65394, ← c65395, k2, ← c65396] at e_1058_3
  have hr := e_1058_3
  simp only [e_1058_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9060)) := by
  have c65397 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65398 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65399 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65400 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65401 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65402 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_4
  simp only [← c65397, ← c65398, ← c65399] at e_1058_4
  have e_1058_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_5
  simp only [← c65400, ← c65401, k2, ← c65402] at e_1058_5
  have hr := e_1058_5
  simp only [e_1058_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9061)) := by
  have c65403 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65404 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65405 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65406 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65407 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65408 := (publicBatchWrapper86_copies2044 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_6
  simp only [← c65403, ← c65404, ← c65405] at e_1058_6
  have e_1058_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_7
  simp only [← c65406, ← c65407, k2, ← c65408] at e_1058_7
  have hr := e_1058_7
  simp only [e_1058_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9062)) := by
  have c65409 := (publicBatchWrapper86_copies2044 a h).2.1
  have c65410 := (publicBatchWrapper86_copies2044 a h).2.2.1
  have c65411 := (publicBatchWrapper86_copies2044 a h).2.2.2.1
  have c65412 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.1
  have c65413 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.1
  have c65414 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_8
  simp only [← c65409, ← c65410, ← c65411] at e_1058_8
  have e_1058_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_9
  simp only [← c65412, ← c65413, k2, ← c65414] at e_1058_9
  have hr := e_1058_9
  simp only [e_1058_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9063)) := by
  have c65415 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.1
  have c65416 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.1
  have c65417 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.1
  have c65418 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65419 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65420 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_10
  simp only [← c65415, ← c65416, ← c65417] at e_1058_10
  have e_1058_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_11
  simp only [← c65418, ← c65419, k2, ← c65420] at e_1058_11
  have hr := e_1058_11
  simp only [e_1058_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9064)) := by
  have c65421 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65422 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65423 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65424 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65425 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65426 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_12
  simp only [← c65421, ← c65422, ← c65423] at e_1058_12
  have e_1058_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_13
  simp only [← c65424, ← c65425, k2, ← c65426] at e_1058_13
  have hr := e_1058_13
  simp only [e_1058_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9065)) := by
  have c65427 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65428 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65429 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65430 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65431 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65432 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_14
  simp only [← c65427, ← c65428, ← c65429] at e_1058_14
  have e_1058_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_15
  simp only [← c65430, ← c65431, k2, ← c65432] at e_1058_15
  have hr := e_1058_15
  simp only [e_1058_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9066)) := by
  have c65433 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65434 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65435 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65436 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65437 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65438 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_16
  simp only [← c65433, ← c65434, ← c65435] at e_1058_16
  have e_1058_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_17
  simp only [← c65436, ← c65437, k2, ← c65438] at e_1058_17
  have hr := e_1058_17
  simp only [e_1058_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9067)) := by
  have c65439 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65440 := (publicBatchWrapper86_copies2045 a h).1
  have c65441 := (publicBatchWrapper86_copies2045 a h).2.1
  have c65442 := (publicBatchWrapper86_copies2045 a h).2.2.1
  have c65443 := (publicBatchWrapper86_copies2045 a h).2.2.2.1
  have c65444 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_18
  simp only [← c65439, ← c65440, ← c65441] at e_1058_18
  have e_1058_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_19
  simp only [← c65442, ← c65443, k2, ← c65444] at e_1058_19
  have hr := e_1058_19
  simp only [e_1058_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9068)) := by
  have c65445 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.1
  have c65446 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.1
  have c65447 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.1
  have c65448 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.1
  have c65449 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.1
  have c65450 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_0
  simp only [← c65445, ← c65446, ← c65447] at e_1059_0
  have e_1059_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_1
  simp only [← c65448, ← c65449, k2, ← c65450] at e_1059_1
  have hr := e_1059_1
  simp only [e_1059_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9069)) := by
  have c65451 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65452 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65453 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65454 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65455 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65456 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_2
  simp only [← c65451, ← c65452, ← c65453] at e_1059_2
  have e_1059_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_3
  simp only [← c65454, ← c65455, k2, ← c65456] at e_1059_3
  have hr := e_1059_3
  simp only [e_1059_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9070)) := by
  have c65457 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65458 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65459 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65460 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65461 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65462 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_4
  simp only [← c65457, ← c65458, ← c65459] at e_1059_4
  have e_1059_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_5
  simp only [← c65460, ← c65461, k2, ← c65462] at e_1059_5
  have hr := e_1059_5
  simp only [e_1059_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9071)) := by
  have c65463 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65464 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65465 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65466 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65467 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65468 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_6
  simp only [← c65463, ← c65464, ← c65465] at e_1059_6
  have e_1059_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_7
  simp only [← c65466, ← c65467, k2, ← c65468] at e_1059_7
  have hr := e_1059_7
  simp only [e_1059_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9072)) := by
  have c65469 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65470 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65471 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65472 := (publicBatchWrapper86_copies2046 a h).1
  have c65473 := (publicBatchWrapper86_copies2046 a h).2.1
  have c65474 := (publicBatchWrapper86_copies2046 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_8
  simp only [← c65469, ← c65470, ← c65471] at e_1059_8
  have e_1059_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_9
  simp only [← c65472, ← c65473, k2, ← c65474] at e_1059_9
  have hr := e_1059_9
  simp only [e_1059_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9073)) := by
  have c65475 := (publicBatchWrapper86_copies2046 a h).2.2.2.1
  have c65476 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.1
  have c65477 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.1
  have c65478 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.1
  have c65479 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.1
  have c65480 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_10
  simp only [← c65475, ← c65476, ← c65477] at e_1059_10
  have e_1059_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_11
  simp only [← c65478, ← c65479, k2, ← c65480] at e_1059_11
  have hr := e_1059_11
  simp only [e_1059_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9074)) := by
  have c65481 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.1
  have c65482 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65483 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65484 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65485 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65486 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_12
  simp only [← c65481, ← c65482, ← c65483] at e_1059_12
  have e_1059_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_13
  simp only [← c65484, ← c65485, k2, ← c65486] at e_1059_13
  have hr := e_1059_13
  simp only [e_1059_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9075)) := by
  have c65487 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65488 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65489 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65490 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65491 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65492 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_14
  simp only [← c65487, ← c65488, ← c65489] at e_1059_14
  have e_1059_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_15
  simp only [← c65490, ← c65491, k2, ← c65492] at e_1059_15
  have hr := e_1059_15
  simp only [e_1059_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9076)) := by
  have c65493 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65494 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65495 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65496 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65497 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65498 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_16
  simp only [← c65493, ← c65494, ← c65495] at e_1059_16
  have e_1059_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_17
  simp only [← c65496, ← c65497, k2, ← c65498] at e_1059_17
  have hr := e_1059_17
  simp only [e_1059_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9077)) := by
  have c65499 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65500 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65501 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65502 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65503 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65504 := (publicBatchWrapper86_copies2047 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_18
  simp only [← c65499, ← c65500, ← c65501] at e_1059_18
  have e_1059_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_19
  simp only [← c65502, ← c65503, k2, ← c65504] at e_1059_19
  have hr := e_1059_19
  simp only [e_1059_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9078)) := by
  have c65505 := (publicBatchWrapper86_copies2047 a h).2.1
  have c65506 := (publicBatchWrapper86_copies2047 a h).2.2.1
  have c65507 := (publicBatchWrapper86_copies2047 a h).2.2.2.1
  have c65508 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.1
  have c65509 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.1
  have c65510 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_0
  simp only [← c65505, ← c65506, ← c65507] at e_1060_0
  have e_1060_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_1
  simp only [← c65508, ← c65509, k2, ← c65510] at e_1060_1
  have hr := e_1060_1
  simp only [e_1060_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9079)) := by
  have c65511 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.1
  have c65512 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.1
  have c65513 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.1
  have c65514 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65515 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65516 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_2
  simp only [← c65511, ← c65512, ← c65513] at e_1060_2
  have e_1060_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_3
  simp only [← c65514, ← c65515, k2, ← c65516] at e_1060_3
  have hr := e_1060_3
  simp only [e_1060_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9080)) := by
  have c65517 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65518 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65519 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65520 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65521 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65522 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_4
  simp only [← c65517, ← c65518, ← c65519] at e_1060_4
  have e_1060_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_5
  simp only [← c65520, ← c65521, k2, ← c65522] at e_1060_5
  have hr := e_1060_5
  simp only [e_1060_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9081)) := by
  have c65523 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65524 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65525 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65526 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65527 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65528 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_6
  simp only [← c65523, ← c65524, ← c65525] at e_1060_6
  have e_1060_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_7
  simp only [← c65526, ← c65527, k2, ← c65528] at e_1060_7
  have hr := e_1060_7
  simp only [e_1060_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9082)) := by
  have c65529 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65530 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65531 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65532 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65533 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65534 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_8
  simp only [← c65529, ← c65530, ← c65531] at e_1060_8
  have e_1060_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_9
  simp only [← c65532, ← c65533, k2, ← c65534] at e_1060_9
  have hr := e_1060_9
  simp only [e_1060_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9083)) := by
  have c65535 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65536 := (publicBatchWrapper86_copies2048 a h).1
  have c65537 := (publicBatchWrapper86_copies2048 a h).2.1
  have c65538 := (publicBatchWrapper86_copies2048 a h).2.2.1
  have c65539 := (publicBatchWrapper86_copies2048 a h).2.2.2.1
  have c65540 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_10
  simp only [← c65535, ← c65536, ← c65537] at e_1060_10
  have e_1060_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_11
  simp only [← c65538, ← c65539, k2, ← c65540] at e_1060_11
  have hr := e_1060_11
  simp only [e_1060_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9084)) := by
  have c65541 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.1
  have c65542 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.1
  have c65543 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.1
  have c65544 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.1
  have c65545 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.1
  have c65546 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_12
  simp only [← c65541, ← c65542, ← c65543] at e_1060_12
  have e_1060_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_13
  simp only [← c65544, ← c65545, k2, ← c65546] at e_1060_13
  have hr := e_1060_13
  simp only [e_1060_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9085)) := by
  have c65547 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65548 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65549 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65550 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65551 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65552 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_14
  simp only [← c65547, ← c65548, ← c65549] at e_1060_14
  have e_1060_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_15
  simp only [← c65550, ← c65551, k2, ← c65552] at e_1060_15
  have hr := e_1060_15
  simp only [e_1060_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9086)) := by
  have c65553 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65554 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65555 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65556 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65557 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65558 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_16
  simp only [← c65553, ← c65554, ← c65555] at e_1060_16
  have e_1060_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_17
  simp only [← c65556, ← c65557, k2, ← c65558] at e_1060_17
  have hr := e_1060_17
  simp only [e_1060_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9087)) := by
  have c65559 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65560 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65561 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65562 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65563 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65564 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_18
  simp only [← c65559, ← c65560, ← c65561] at e_1060_18
  have e_1060_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_19
  simp only [← c65562, ← c65563, k2, ← c65564] at e_1060_19
  have hr := e_1060_19
  simp only [e_1060_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9088)) := by
  have c65565 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65566 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65567 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65568 := (publicBatchWrapper86_copies2049 a h).1
  have c65569 := (publicBatchWrapper86_copies2049 a h).2.1
  have c65570 := (publicBatchWrapper86_copies2049 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_0
  simp only [← c65565, ← c65566, ← c65567] at e_1061_0
  have e_1061_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_1
  simp only [← c65568, ← c65569, k2, ← c65570] at e_1061_1
  have hr := e_1061_1
  simp only [e_1061_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9089)) := by
  have c65571 := (publicBatchWrapper86_copies2049 a h).2.2.2.1
  have c65572 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.1
  have c65573 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.1
  have c65574 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.1
  have c65575 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.1
  have c65576 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_2
  simp only [← c65571, ← c65572, ← c65573] at e_1061_2
  have e_1061_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_3
  simp only [← c65574, ← c65575, k2, ← c65576] at e_1061_3
  have hr := e_1061_3
  simp only [e_1061_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9090)) := by
  have c65577 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.1
  have c65578 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65579 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65580 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65581 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65582 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_4
  simp only [← c65577, ← c65578, ← c65579] at e_1061_4
  have e_1061_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_5
  simp only [← c65580, ← c65581, k2, ← c65582] at e_1061_5
  have hr := e_1061_5
  simp only [e_1061_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9091)) := by
  have c65583 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65584 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65585 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65586 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65587 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65588 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_6
  simp only [← c65583, ← c65584, ← c65585] at e_1061_6
  have e_1061_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_7
  simp only [← c65586, ← c65587, k2, ← c65588] at e_1061_7
  have hr := e_1061_7
  simp only [e_1061_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9092)) := by
  have c65589 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65590 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65591 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65592 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65593 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65594 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_8
  simp only [← c65589, ← c65590, ← c65591] at e_1061_8
  have e_1061_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_9
  simp only [← c65592, ← c65593, k2, ← c65594] at e_1061_9
  have hr := e_1061_9
  simp only [e_1061_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9093)) := by
  have c65595 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65596 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65597 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65598 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65599 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65600 := (publicBatchWrapper86_copies2050 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_10
  simp only [← c65595, ← c65596, ← c65597] at e_1061_10
  have e_1061_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_11
  simp only [← c65598, ← c65599, k2, ← c65600] at e_1061_11
  have hr := e_1061_11
  simp only [e_1061_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9094)) := by
  have c65601 := (publicBatchWrapper86_copies2050 a h).2.1
  have c65602 := (publicBatchWrapper86_copies2050 a h).2.2.1
  have c65603 := (publicBatchWrapper86_copies2050 a h).2.2.2.1
  have c65604 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.1
  have c65605 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.1
  have c65606 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_12
  simp only [← c65601, ← c65602, ← c65603] at e_1061_12
  have e_1061_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_13
  simp only [← c65604, ← c65605, k2, ← c65606] at e_1061_13
  have hr := e_1061_13
  simp only [e_1061_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9095)) := by
  have c65607 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.1
  have c65608 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.1
  have c65609 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.1
  have c65610 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65611 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65612 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_14
  simp only [← c65607, ← c65608, ← c65609] at e_1061_14
  have e_1061_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_15
  simp only [← c65610, ← c65611, k2, ← c65612] at e_1061_15
  have hr := e_1061_15
  simp only [e_1061_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9096)) := by
  have c65613 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65614 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65615 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65616 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65617 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65618 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_16
  simp only [← c65613, ← c65614, ← c65615] at e_1061_16
  have e_1061_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_17
  simp only [← c65616, ← c65617, k2, ← c65618] at e_1061_17
  have hr := e_1061_17
  simp only [e_1061_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9097)) := by
  have c65619 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65620 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65621 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65622 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65623 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65624 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_18
  simp only [← c65619, ← c65620, ← c65621] at e_1061_18
  have e_1061_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_19
  simp only [← c65622, ← c65623, k2, ← c65624] at e_1061_19
  have hr := e_1061_19
  simp only [e_1061_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9098)) := by
  have c65625 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65626 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65627 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65628 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65629 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65630 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_0
  simp only [← c65625, ← c65626, ← c65627] at e_1062_0
  have e_1062_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_1
  simp only [← c65628, ← c65629, k2, ← c65630] at e_1062_1
  have hr := e_1062_1
  simp only [e_1062_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9099)) := by
  have c65631 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65632 := (publicBatchWrapper86_copies2051 a h).1
  have c65633 := (publicBatchWrapper86_copies2051 a h).2.1
  have c65634 := (publicBatchWrapper86_copies2051 a h).2.2.1
  have c65635 := (publicBatchWrapper86_copies2051 a h).2.2.2.1
  have c65636 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_2
  simp only [← c65631, ← c65632, ← c65633] at e_1062_2
  have e_1062_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_3
  simp only [← c65634, ← c65635, k2, ← c65636] at e_1062_3
  have hr := e_1062_3
  simp only [e_1062_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9100)) := by
  have c65637 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.1
  have c65638 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.1
  have c65639 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.1
  have c65640 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.1
  have c65641 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.1
  have c65642 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_4
  simp only [← c65637, ← c65638, ← c65639] at e_1062_4
  have e_1062_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_5
  simp only [← c65640, ← c65641, k2, ← c65642] at e_1062_5
  have hr := e_1062_5
  simp only [e_1062_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9101)) := by
  have c65643 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65644 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65645 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65646 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65647 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65648 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_6
  simp only [← c65643, ← c65644, ← c65645] at e_1062_6
  have e_1062_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_7
  simp only [← c65646, ← c65647, k2, ← c65648] at e_1062_7
  have hr := e_1062_7
  simp only [e_1062_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9102)) := by
  have c65649 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65650 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65651 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65652 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65653 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65654 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_8
  simp only [← c65649, ← c65650, ← c65651] at e_1062_8
  have e_1062_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_9
  simp only [← c65652, ← c65653, k2, ← c65654] at e_1062_9
  have hr := e_1062_9
  simp only [e_1062_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9103)) := by
  have c65655 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65656 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65657 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65658 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65659 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65660 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_10
  simp only [← c65655, ← c65656, ← c65657] at e_1062_10
  have e_1062_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_11
  simp only [← c65658, ← c65659, k2, ← c65660] at e_1062_11
  have hr := e_1062_11
  simp only [e_1062_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9104)) := by
  have c65661 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65662 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65663 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65664 := (publicBatchWrapper86_copies2052 a h).1
  have c65665 := (publicBatchWrapper86_copies2052 a h).2.1
  have c65666 := (publicBatchWrapper86_copies2052 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_12
  simp only [← c65661, ← c65662, ← c65663] at e_1062_12
  have e_1062_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_13
  simp only [← c65664, ← c65665, k2, ← c65666] at e_1062_13
  have hr := e_1062_13
  simp only [e_1062_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9105)) := by
  have c65667 := (publicBatchWrapper86_copies2052 a h).2.2.2.1
  have c65668 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.1
  have c65669 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.1
  have c65670 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.1
  have c65671 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.1
  have c65672 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_14
  simp only [← c65667, ← c65668, ← c65669] at e_1062_14
  have e_1062_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_15
  simp only [← c65670, ← c65671, k2, ← c65672] at e_1062_15
  have hr := e_1062_15
  simp only [e_1062_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9106)) := by
  have c65673 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.1
  have c65674 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65675 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65676 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65677 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65678 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_16
  simp only [← c65673, ← c65674, ← c65675] at e_1062_16
  have e_1062_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_17
  simp only [← c65676, ← c65677, k2, ← c65678] at e_1062_17
  have hr := e_1062_17
  simp only [e_1062_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9107)) := by
  have c65679 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65680 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65681 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65682 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65683 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65684 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_18
  simp only [← c65679, ← c65680, ← c65681] at e_1062_18
  have e_1062_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_19
  simp only [← c65682, ← c65683, k2, ← c65684] at e_1062_19
  have hr := e_1062_19
  simp only [e_1062_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9108)) := by
  have c65685 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65686 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65687 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65688 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65689 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65690 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_0
  simp only [← c65685, ← c65686, ← c65687] at e_1063_0
  have e_1063_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_1
  simp only [← c65688, ← c65689, k2, ← c65690] at e_1063_1
  have hr := e_1063_1
  simp only [e_1063_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9109)) := by
  have c65691 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65692 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65693 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65694 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65695 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65696 := (publicBatchWrapper86_copies2053 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_2
  simp only [← c65691, ← c65692, ← c65693] at e_1063_2
  have e_1063_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_3
  simp only [← c65694, ← c65695, k2, ← c65696] at e_1063_3
  have hr := e_1063_3
  simp only [e_1063_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9110)) := by
  have c65697 := (publicBatchWrapper86_copies2053 a h).2.1
  have c65698 := (publicBatchWrapper86_copies2053 a h).2.2.1
  have c65699 := (publicBatchWrapper86_copies2053 a h).2.2.2.1
  have c65700 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.1
  have c65701 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.1
  have c65702 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_4
  simp only [← c65697, ← c65698, ← c65699] at e_1063_4
  have e_1063_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_5
  simp only [← c65700, ← c65701, k2, ← c65702] at e_1063_5
  have hr := e_1063_5
  simp only [e_1063_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9111)) := by
  have c65703 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.1
  have c65704 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.1
  have c65705 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.1
  have c65706 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65707 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65708 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_6
  simp only [← c65703, ← c65704, ← c65705] at e_1063_6
  have e_1063_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_7
  simp only [← c65706, ← c65707, k2, ← c65708] at e_1063_7
  have hr := e_1063_7
  simp only [e_1063_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9112)) := by
  have c65709 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65710 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65711 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65712 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65713 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65714 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_8
  simp only [← c65709, ← c65710, ← c65711] at e_1063_8
  have e_1063_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_9
  simp only [← c65712, ← c65713, k2, ← c65714] at e_1063_9
  have hr := e_1063_9
  simp only [e_1063_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9113)) := by
  have c65715 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65716 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65717 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65718 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65719 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65720 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_10
  simp only [← c65715, ← c65716, ← c65717] at e_1063_10
  have e_1063_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_11
  simp only [← c65718, ← c65719, k2, ← c65720] at e_1063_11
  have hr := e_1063_11
  simp only [e_1063_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9114)) := by
  have c65721 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65722 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65723 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65724 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65725 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65726 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_12
  simp only [← c65721, ← c65722, ← c65723] at e_1063_12
  have e_1063_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_13
  simp only [← c65724, ← c65725, k2, ← c65726] at e_1063_13
  have hr := e_1063_13
  simp only [e_1063_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9115)) := by
  have c65727 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65728 := (publicBatchWrapper86_copies2054 a h).1
  have c65729 := (publicBatchWrapper86_copies2054 a h).2.1
  have c65730 := (publicBatchWrapper86_copies2054 a h).2.2.1
  have c65731 := (publicBatchWrapper86_copies2054 a h).2.2.2.1
  have c65732 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_14
  simp only [← c65727, ← c65728, ← c65729] at e_1063_14
  have e_1063_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_15
  simp only [← c65730, ← c65731, k2, ← c65732] at e_1063_15
  have hr := e_1063_15
  simp only [e_1063_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9116)) := by
  have c65733 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.1
  have c65734 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.1
  have c65735 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.1
  have c65736 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.1
  have c65737 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.1
  have c65738 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_16
  simp only [← c65733, ← c65734, ← c65735] at e_1063_16
  have e_1063_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_17
  simp only [← c65736, ← c65737, k2, ← c65738] at e_1063_17
  have hr := e_1063_17
  simp only [e_1063_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9117)) := by
  have c65739 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65740 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65741 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65742 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65743 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65744 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_18
  simp only [← c65739, ← c65740, ← c65741] at e_1063_18
  have e_1063_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_19
  simp only [← c65742, ← c65743, k2, ← c65744] at e_1063_19
  have hr := e_1063_19
  simp only [e_1063_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9118)) := by
  have c65745 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65746 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65747 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65748 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65749 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65750 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_0
  simp only [← c65745, ← c65746, ← c65747] at e_1064_0
  have e_1064_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_1
  simp only [← c65748, ← c65749, k2, ← c65750] at e_1064_1
  have hr := e_1064_1
  simp only [e_1064_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9119)) := by
  have c65751 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65752 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65753 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65754 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65755 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65756 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_2
  simp only [← c65751, ← c65752, ← c65753] at e_1064_2
  have e_1064_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_3
  simp only [← c65754, ← c65755, k2, ← c65756] at e_1064_3
  have hr := e_1064_3
  simp only [e_1064_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9120)) := by
  have c65757 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65758 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65759 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65760 := (publicBatchWrapper86_copies2055 a h).1
  have c65761 := (publicBatchWrapper86_copies2055 a h).2.1
  have c65762 := (publicBatchWrapper86_copies2055 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_4
  simp only [← c65757, ← c65758, ← c65759] at e_1064_4
  have e_1064_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_5
  simp only [← c65760, ← c65761, k2, ← c65762] at e_1064_5
  have hr := e_1064_5
  simp only [e_1064_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9121)) := by
  have c65763 := (publicBatchWrapper86_copies2055 a h).2.2.2.1
  have c65764 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.1
  have c65765 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.1
  have c65766 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.1
  have c65767 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.1
  have c65768 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_6
  simp only [← c65763, ← c65764, ← c65765] at e_1064_6
  have e_1064_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_7
  simp only [← c65766, ← c65767, k2, ← c65768] at e_1064_7
  have hr := e_1064_7
  simp only [e_1064_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9122)) := by
  have c65769 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.1
  have c65770 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65771 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65772 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65773 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65774 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_8
  simp only [← c65769, ← c65770, ← c65771] at e_1064_8
  have e_1064_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_9
  simp only [← c65772, ← c65773, k2, ← c65774] at e_1064_9
  have hr := e_1064_9
  simp only [e_1064_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9123)) := by
  have c65775 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65776 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65777 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65778 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65779 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65780 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_10
  simp only [← c65775, ← c65776, ← c65777] at e_1064_10
  have e_1064_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_11
  simp only [← c65778, ← c65779, k2, ← c65780] at e_1064_11
  have hr := e_1064_11
  simp only [e_1064_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9124)) := by
  have c65781 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65782 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65783 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65784 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65785 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65786 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_12
  simp only [← c65781, ← c65782, ← c65783] at e_1064_12
  have e_1064_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_13
  simp only [← c65784, ← c65785, k2, ← c65786] at e_1064_13
  have hr := e_1064_13
  simp only [e_1064_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9125)) := by
  have c65787 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65788 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65789 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65790 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65791 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65792 := (publicBatchWrapper86_copies2056 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_14
  simp only [← c65787, ← c65788, ← c65789] at e_1064_14
  have e_1064_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_15
  simp only [← c65790, ← c65791, k2, ← c65792] at e_1064_15
  have hr := e_1064_15
  simp only [e_1064_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9126)) := by
  have c65793 := (publicBatchWrapper86_copies2056 a h).2.1
  have c65794 := (publicBatchWrapper86_copies2056 a h).2.2.1
  have c65795 := (publicBatchWrapper86_copies2056 a h).2.2.2.1
  have c65796 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.1
  have c65797 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.1
  have c65798 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_16
  simp only [← c65793, ← c65794, ← c65795] at e_1064_16
  have e_1064_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_17
  simp only [← c65796, ← c65797, k2, ← c65798] at e_1064_17
  have hr := e_1064_17
  simp only [e_1064_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9127)) := by
  have c65799 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.1
  have c65800 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.1
  have c65801 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.1
  have c65802 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65803 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65804 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_18
  simp only [← c65799, ← c65800, ← c65801] at e_1064_18
  have e_1064_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_19
  simp only [← c65802, ← c65803, k2, ← c65804] at e_1064_19
  have hr := e_1064_19
  simp only [e_1064_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9128)) := by
  have c65805 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65806 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65807 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65808 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65809 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65810 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_0
  simp only [← c65805, ← c65806, ← c65807] at e_1065_0
  have e_1065_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_1
  simp only [← c65808, ← c65809, k2, ← c65810] at e_1065_1
  have hr := e_1065_1
  simp only [e_1065_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9129)) := by
  have c65811 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65812 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65813 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65814 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65815 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65816 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_2
  simp only [← c65811, ← c65812, ← c65813] at e_1065_2
  have e_1065_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_3
  simp only [← c65814, ← c65815, k2, ← c65816] at e_1065_3
  have hr := e_1065_3
  simp only [e_1065_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9130)) := by
  have c65817 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65818 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65819 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65820 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65821 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65822 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_4
  simp only [← c65817, ← c65818, ← c65819] at e_1065_4
  have e_1065_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_5
  simp only [← c65820, ← c65821, k2, ← c65822] at e_1065_5
  have hr := e_1065_5
  simp only [e_1065_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9131)) := by
  have c65823 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65824 := (publicBatchWrapper86_copies2057 a h).1
  have c65825 := (publicBatchWrapper86_copies2057 a h).2.1
  have c65826 := (publicBatchWrapper86_copies2057 a h).2.2.1
  have c65827 := (publicBatchWrapper86_copies2057 a h).2.2.2.1
  have c65828 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_6
  simp only [← c65823, ← c65824, ← c65825] at e_1065_6
  have e_1065_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_7
  simp only [← c65826, ← c65827, k2, ← c65828] at e_1065_7
  have hr := e_1065_7
  simp only [e_1065_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9132)) := by
  have c65829 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.1
  have c65830 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.1
  have c65831 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.1
  have c65832 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.1
  have c65833 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.1
  have c65834 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_8
  simp only [← c65829, ← c65830, ← c65831] at e_1065_8
  have e_1065_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_9
  simp only [← c65832, ← c65833, k2, ← c65834] at e_1065_9
  have hr := e_1065_9
  simp only [e_1065_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9133)) := by
  have c65835 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65836 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65837 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65838 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65839 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65840 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_10
  simp only [← c65835, ← c65836, ← c65837] at e_1065_10
  have e_1065_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_11
  simp only [← c65838, ← c65839, k2, ← c65840] at e_1065_11
  have hr := e_1065_11
  simp only [e_1065_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9134)) := by
  have c65841 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65842 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65843 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65844 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65845 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65846 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_12
  simp only [← c65841, ← c65842, ← c65843] at e_1065_12
  have e_1065_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_13
  simp only [← c65844, ← c65845, k2, ← c65846] at e_1065_13
  have hr := e_1065_13
  simp only [e_1065_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9135)) := by
  have c65847 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65848 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65849 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65850 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65851 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65852 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_14
  simp only [← c65847, ← c65848, ← c65849] at e_1065_14
  have e_1065_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_15
  simp only [← c65850, ← c65851, k2, ← c65852] at e_1065_15
  have hr := e_1065_15
  simp only [e_1065_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9136)) := by
  have c65853 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65854 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65855 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65856 := (publicBatchWrapper86_copies2058 a h).1
  have c65857 := (publicBatchWrapper86_copies2058 a h).2.1
  have c65858 := (publicBatchWrapper86_copies2058 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_16
  simp only [← c65853, ← c65854, ← c65855] at e_1065_16
  have e_1065_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_17
  simp only [← c65856, ← c65857, k2, ← c65858] at e_1065_17
  have hr := e_1065_17
  simp only [e_1065_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9137)) := by
  have c65859 := (publicBatchWrapper86_copies2058 a h).2.2.2.1
  have c65860 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.1
  have c65861 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.1
  have c65862 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.1
  have c65863 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.1
  have c65864 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_18
  simp only [← c65859, ← c65860, ← c65861] at e_1065_18
  have e_1065_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_19
  simp only [← c65862, ← c65863, k2, ← c65864] at e_1065_19
  have hr := e_1065_19
  simp only [e_1065_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9138)) := by
  have c65865 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.1
  have c65866 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65867 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65868 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65869 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65870 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_0
  simp only [← c65865, ← c65866, ← c65867] at e_1066_0
  have e_1066_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_1
  simp only [← c65868, ← c65869, k2, ← c65870] at e_1066_1
  have hr := e_1066_1
  simp only [e_1066_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9139)) := by
  have c65871 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65872 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65873 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65874 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65875 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65876 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_2
  simp only [← c65871, ← c65872, ← c65873] at e_1066_2
  have e_1066_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_3
  simp only [← c65874, ← c65875, k2, ← c65876] at e_1066_3
  have hr := e_1066_3
  simp only [e_1066_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9140)) := by
  have c65877 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65878 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65879 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65880 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65881 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65882 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_4
  simp only [← c65877, ← c65878, ← c65879] at e_1066_4
  have e_1066_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_5
  simp only [← c65880, ← c65881, k2, ← c65882] at e_1066_5
  have hr := e_1066_5
  simp only [e_1066_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9141)) := by
  have c65883 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65884 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65885 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65886 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65887 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65888 := (publicBatchWrapper86_copies2059 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_6
  simp only [← c65883, ← c65884, ← c65885] at e_1066_6
  have e_1066_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_7
  simp only [← c65886, ← c65887, k2, ← c65888] at e_1066_7
  have hr := e_1066_7
  simp only [e_1066_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9142)) := by
  have c65889 := (publicBatchWrapper86_copies2059 a h).2.1
  have c65890 := (publicBatchWrapper86_copies2059 a h).2.2.1
  have c65891 := (publicBatchWrapper86_copies2059 a h).2.2.2.1
  have c65892 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.1
  have c65893 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.1
  have c65894 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_8
  simp only [← c65889, ← c65890, ← c65891] at e_1066_8
  have e_1066_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_9
  simp only [← c65892, ← c65893, k2, ← c65894] at e_1066_9
  have hr := e_1066_9
  simp only [e_1066_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9143)) := by
  have c65895 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.1
  have c65896 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.1
  have c65897 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.1
  have c65898 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65899 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65900 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_10
  simp only [← c65895, ← c65896, ← c65897] at e_1066_10
  have e_1066_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_11
  simp only [← c65898, ← c65899, k2, ← c65900] at e_1066_11
  have hr := e_1066_11
  simp only [e_1066_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9144)) := by
  have c65901 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65902 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65903 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65904 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65905 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65906 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_12
  simp only [← c65901, ← c65902, ← c65903] at e_1066_12
  have e_1066_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_13
  simp only [← c65904, ← c65905, k2, ← c65906] at e_1066_13
  have hr := e_1066_13
  simp only [e_1066_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9145)) := by
  have c65907 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65908 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65909 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65910 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65911 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65912 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_14
  simp only [← c65907, ← c65908, ← c65909] at e_1066_14
  have e_1066_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_15
  simp only [← c65910, ← c65911, k2, ← c65912] at e_1066_15
  have hr := e_1066_15
  simp only [e_1066_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9146)) := by
  have c65913 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65914 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65915 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65916 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65917 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65918 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_16
  simp only [← c65913, ← c65914, ← c65915] at e_1066_16
  have e_1066_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_17
  simp only [← c65916, ← c65917, k2, ← c65918] at e_1066_17
  have hr := e_1066_17
  simp only [e_1066_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9147)) := by
  have c65919 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65920 := (publicBatchWrapper86_copies2060 a h).1
  have c65921 := (publicBatchWrapper86_copies2060 a h).2.1
  have c65922 := (publicBatchWrapper86_copies2060 a h).2.2.1
  have c65923 := (publicBatchWrapper86_copies2060 a h).2.2.2.1
  have c65924 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_18
  simp only [← c65919, ← c65920, ← c65921] at e_1066_18
  have e_1066_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_19
  simp only [← c65922, ← c65923, k2, ← c65924] at e_1066_19
  have hr := e_1066_19
  simp only [e_1066_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9148)) := by
  have c65925 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.1
  have c65926 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.1
  have c65927 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.1
  have c65928 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.1
  have c65929 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.1
  have c65930 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_0
  simp only [← c65925, ← c65926, ← c65927] at e_1067_0
  have e_1067_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_1
  simp only [← c65928, ← c65929, k2, ← c65930] at e_1067_1
  have hr := e_1067_1
  simp only [e_1067_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9149)) := by
  have c65931 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65932 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65933 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65934 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65935 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65936 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_2
  simp only [← c65931, ← c65932, ← c65933] at e_1067_2
  have e_1067_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_3
  simp only [← c65934, ← c65935, k2, ← c65936] at e_1067_3
  have hr := e_1067_3
  simp only [e_1067_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9150)) := by
  have c65937 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65938 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65939 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65940 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65941 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65942 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_4
  simp only [← c65937, ← c65938, ← c65939] at e_1067_4
  have e_1067_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_5
  simp only [← c65940, ← c65941, k2, ← c65942] at e_1067_5
  have hr := e_1067_5
  simp only [e_1067_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9151)) := by
  have c65943 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65944 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65945 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65946 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65947 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65948 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_6
  simp only [← c65943, ← c65944, ← c65945] at e_1067_6
  have e_1067_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_7
  simp only [← c65946, ← c65947, k2, ← c65948] at e_1067_7
  have hr := e_1067_7
  simp only [e_1067_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9152)) := by
  have c65949 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65950 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65951 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65952 := (publicBatchWrapper86_copies2061 a h).1
  have c65953 := (publicBatchWrapper86_copies2061 a h).2.1
  have c65954 := (publicBatchWrapper86_copies2061 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_8
  simp only [← c65949, ← c65950, ← c65951] at e_1067_8
  have e_1067_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_9
  simp only [← c65952, ← c65953, k2, ← c65954] at e_1067_9
  have hr := e_1067_9
  simp only [e_1067_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9153)) := by
  have c65955 := (publicBatchWrapper86_copies2061 a h).2.2.2.1
  have c65956 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.1
  have c65957 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.1
  have c65958 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.1
  have c65959 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.1
  have c65960 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_10
  simp only [← c65955, ← c65956, ← c65957] at e_1067_10
  have e_1067_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_11
  simp only [← c65958, ← c65959, k2, ← c65960] at e_1067_11
  have hr := e_1067_11
  simp only [e_1067_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9154)) := by
  have c65961 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.1
  have c65962 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65963 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65964 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65965 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65966 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_12
  simp only [← c65961, ← c65962, ← c65963] at e_1067_12
  have e_1067_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_13
  simp only [← c65964, ← c65965, k2, ← c65966] at e_1067_13
  have hr := e_1067_13
  simp only [e_1067_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9155)) := by
  have c65967 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65968 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65969 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65970 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65971 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65972 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_14
  simp only [← c65967, ← c65968, ← c65969] at e_1067_14
  have e_1067_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_15
  simp only [← c65970, ← c65971, k2, ← c65972] at e_1067_15
  have hr := e_1067_15
  simp only [e_1067_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9156)) := by
  have c65973 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65974 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65975 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65976 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65977 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65978 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_16
  simp only [← c65973, ← c65974, ← c65975] at e_1067_16
  have e_1067_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_17
  simp only [← c65976, ← c65977, k2, ← c65978] at e_1067_17
  have hr := e_1067_17
  simp only [e_1067_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9157)) := by
  have c65979 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65980 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65981 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65982 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65983 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65984 := (publicBatchWrapper86_copies2062 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_18
  simp only [← c65979, ← c65980, ← c65981] at e_1067_18
  have e_1067_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_19
  simp only [← c65982, ← c65983, k2, ← c65984] at e_1067_19
  have hr := e_1067_19
  simp only [e_1067_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9158)) := by
  have c65985 := (publicBatchWrapper86_copies2062 a h).2.1
  have c65986 := (publicBatchWrapper86_copies2062 a h).2.2.1
  have c65987 := (publicBatchWrapper86_copies2062 a h).2.2.2.1
  have c65988 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.1
  have c65989 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.1
  have c65990 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_0
  simp only [← c65985, ← c65986, ← c65987] at e_1068_0
  have e_1068_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_1
  simp only [← c65988, ← c65989, k2, ← c65990] at e_1068_1
  have hr := e_1068_1
  simp only [e_1068_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9159)) := by
  have c65991 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.1
  have c65992 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.1
  have c65993 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.1
  have c65994 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65995 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65996 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_2
  simp only [← c65991, ← c65992, ← c65993] at e_1068_2
  have e_1068_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_3
  simp only [← c65994, ← c65995, k2, ← c65996] at e_1068_3
  have hr := e_1068_3
  simp only [e_1068_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9160)) := by
  have c65997 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65998 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65999 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66000 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66001 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66002 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_4
  simp only [← c65997, ← c65998, ← c65999] at e_1068_4
  have e_1068_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_5
  simp only [← c66000, ← c66001, k2, ← c66002] at e_1068_5
  have hr := e_1068_5
  simp only [e_1068_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9161)) := by
  have c66003 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66004 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66005 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66006 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66007 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66008 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_6
  simp only [← c66003, ← c66004, ← c66005] at e_1068_6
  have e_1068_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_7
  simp only [← c66006, ← c66007, k2, ← c66008] at e_1068_7
  have hr := e_1068_7
  simp only [e_1068_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9162)) := by
  have c66009 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66010 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66011 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66012 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66013 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66014 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_8
  simp only [← c66009, ← c66010, ← c66011] at e_1068_8
  have e_1068_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_9
  simp only [← c66012, ← c66013, k2, ← c66014] at e_1068_9
  have hr := e_1068_9
  simp only [e_1068_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9163)) := by
  have c66015 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66016 := (publicBatchWrapper86_copies2063 a h).1
  have c66017 := (publicBatchWrapper86_copies2063 a h).2.1
  have c66018 := (publicBatchWrapper86_copies2063 a h).2.2.1
  have c66019 := (publicBatchWrapper86_copies2063 a h).2.2.2.1
  have c66020 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_10
  simp only [← c66015, ← c66016, ← c66017] at e_1068_10
  have e_1068_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_11
  simp only [← c66018, ← c66019, k2, ← c66020] at e_1068_11
  have hr := e_1068_11
  simp only [e_1068_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9164)) := by
  have c66021 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.1
  have c66022 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.1
  have c66023 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.1
  have c66024 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.1
  have c66025 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.1
  have c66026 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_12
  simp only [← c66021, ← c66022, ← c66023] at e_1068_12
  have e_1068_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_13
  simp only [← c66024, ← c66025, k2, ← c66026] at e_1068_13
  have hr := e_1068_13
  simp only [e_1068_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9165)) := by
  have c66027 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66028 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66029 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66030 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66031 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66032 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_14
  simp only [← c66027, ← c66028, ← c66029] at e_1068_14
  have e_1068_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_15
  simp only [← c66030, ← c66031, k2, ← c66032] at e_1068_15
  have hr := e_1068_15
  simp only [e_1068_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9166)) := by
  have c66033 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66034 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66035 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66036 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66037 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66038 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_16
  simp only [← c66033, ← c66034, ← c66035] at e_1068_16
  have e_1068_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_17
  simp only [← c66036, ← c66037, k2, ← c66038] at e_1068_17
  have hr := e_1068_17
  simp only [e_1068_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9167)) := by
  have c66039 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66040 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66041 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66042 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66043 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66044 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_18
  simp only [← c66039, ← c66040, ← c66041] at e_1068_18
  have e_1068_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_19
  simp only [← c66042, ← c66043, k2, ← c66044] at e_1068_19
  have hr := e_1068_19
  simp only [e_1068_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9168)) := by
  have c66045 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66046 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66047 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66048 := (publicBatchWrapper86_copies2064 a h).1
  have c66049 := (publicBatchWrapper86_copies2064 a h).2.1
  have c66050 := (publicBatchWrapper86_copies2064 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_0
  simp only [← c66045, ← c66046, ← c66047] at e_1069_0
  have e_1069_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_1
  simp only [← c66048, ← c66049, k2, ← c66050] at e_1069_1
  have hr := e_1069_1
  simp only [e_1069_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9169)) := by
  have c66051 := (publicBatchWrapper86_copies2064 a h).2.2.2.1
  have c66052 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.1
  have c66053 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.1
  have c66054 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.1
  have c66055 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.1
  have c66056 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_2
  simp only [← c66051, ← c66052, ← c66053] at e_1069_2
  have e_1069_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_3
  simp only [← c66054, ← c66055, k2, ← c66056] at e_1069_3
  have hr := e_1069_3
  simp only [e_1069_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9210)) := by
  have c66057 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.1
  have c66058 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66059 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66060 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66061 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66062 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_4
  simp only [← c66057, ← c66058, ← c66059] at e_1069_4
  have e_1069_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_5
  simp only [← c66060, ← c66061, k2, ← c66062] at e_1069_5
  have hr := e_1069_5
  simp only [e_1069_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9211)) := by
  have c66063 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66064 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66065 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66066 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66067 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66068 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_6
  simp only [← c66063, ← c66064, ← c66065] at e_1069_6
  have e_1069_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_7
  simp only [← c66066, ← c66067, k2, ← c66068] at e_1069_7
  have hr := e_1069_7
  simp only [e_1069_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9212)) := by
  have c66069 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66070 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66071 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66072 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66073 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66074 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_8
  simp only [← c66069, ← c66070, ← c66071] at e_1069_8
  have e_1069_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_9
  simp only [← c66072, ← c66073, k2, ← c66074] at e_1069_9
  have hr := e_1069_9
  simp only [e_1069_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9213)) := by
  have c66075 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66076 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66077 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66078 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66079 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66080 := (publicBatchWrapper86_copies2065 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_10
  simp only [← c66075, ← c66076, ← c66077] at e_1069_10
  have e_1069_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_11
  simp only [← c66078, ← c66079, k2, ← c66080] at e_1069_11
  have hr := e_1069_11
  simp only [e_1069_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9214)) := by
  have c66081 := (publicBatchWrapper86_copies2065 a h).2.1
  have c66082 := (publicBatchWrapper86_copies2065 a h).2.2.1
  have c66083 := (publicBatchWrapper86_copies2065 a h).2.2.2.1
  have c66084 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.1
  have c66085 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.1
  have c66086 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_12
  simp only [← c66081, ← c66082, ← c66083] at e_1069_12
  have e_1069_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_13
  simp only [← c66084, ← c66085, k2, ← c66086] at e_1069_13
  have hr := e_1069_13
  simp only [e_1069_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9215)) := by
  have c66087 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.1
  have c66088 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.1
  have c66089 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.1
  have c66090 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66091 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66092 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_14
  simp only [← c66087, ← c66088, ← c66089] at e_1069_14
  have e_1069_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_15
  simp only [← c66090, ← c66091, k2, ← c66092] at e_1069_15
  have hr := e_1069_15
  simp only [e_1069_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9216)) := by
  have c66093 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66094 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66095 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66096 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66097 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66098 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_16
  simp only [← c66093, ← c66094, ← c66095] at e_1069_16
  have e_1069_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_17
  simp only [← c66096, ← c66097, k2, ← c66098] at e_1069_17
  have hr := e_1069_17
  simp only [e_1069_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9217)) := by
  have c66099 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66100 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66101 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66102 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66103 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66104 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_18
  simp only [← c66099, ← c66100, ← c66101] at e_1069_18
  have e_1069_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_19
  simp only [← c66102, ← c66103, k2, ← c66104] at e_1069_19
  have hr := e_1069_19
  simp only [e_1069_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9218)) := by
  have c66105 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66106 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66107 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66108 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66109 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66110 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_0
  simp only [← c66105, ← c66106, ← c66107] at e_1070_0
  have e_1070_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_1
  simp only [← c66108, ← c66109, k2, ← c66110] at e_1070_1
  have hr := e_1070_1
  simp only [e_1070_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9219)) := by
  have c66111 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66112 := (publicBatchWrapper86_copies2066 a h).1
  have c66113 := (publicBatchWrapper86_copies2066 a h).2.1
  have c66114 := (publicBatchWrapper86_copies2066 a h).2.2.1
  have c66115 := (publicBatchWrapper86_copies2066 a h).2.2.2.1
  have c66116 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_2
  simp only [← c66111, ← c66112, ← c66113] at e_1070_2
  have e_1070_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_3
  simp only [← c66114, ← c66115, k2, ← c66116] at e_1070_3
  have hr := e_1070_3
  simp only [e_1070_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9220)) := by
  have c66117 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.1
  have c66118 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.1
  have c66119 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.1
  have c66120 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.1
  have c66121 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.1
  have c66122 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_4
  simp only [← c66117, ← c66118, ← c66119] at e_1070_4
  have e_1070_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_5
  simp only [← c66120, ← c66121, k2, ← c66122] at e_1070_5
  have hr := e_1070_5
  simp only [e_1070_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9221)) := by
  have c66123 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66124 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66125 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66126 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66127 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66128 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_6
  simp only [← c66123, ← c66124, ← c66125] at e_1070_6
  have e_1070_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_7
  simp only [← c66126, ← c66127, k2, ← c66128] at e_1070_7
  have hr := e_1070_7
  simp only [e_1070_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9222)) := by
  have c66129 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66130 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66131 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66132 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66133 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66134 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_8
  simp only [← c66129, ← c66130, ← c66131] at e_1070_8
  have e_1070_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_9
  simp only [← c66132, ← c66133, k2, ← c66134] at e_1070_9
  have hr := e_1070_9
  simp only [e_1070_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9223)) := by
  have c66135 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66136 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66137 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66138 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66139 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66140 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_10
  simp only [← c66135, ← c66136, ← c66137] at e_1070_10
  have e_1070_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_11
  simp only [← c66138, ← c66139, k2, ← c66140] at e_1070_11
  have hr := e_1070_11
  simp only [e_1070_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9224)) := by
  have c66141 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66142 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66143 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66144 := (publicBatchWrapper86_copies2067 a h).1
  have c66145 := (publicBatchWrapper86_copies2067 a h).2.1
  have c66146 := (publicBatchWrapper86_copies2067 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_12
  simp only [← c66141, ← c66142, ← c66143] at e_1070_12
  have e_1070_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_13
  simp only [← c66144, ← c66145, k2, ← c66146] at e_1070_13
  have hr := e_1070_13
  simp only [e_1070_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9225)) := by
  have c66147 := (publicBatchWrapper86_copies2067 a h).2.2.2.1
  have c66148 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.1
  have c66149 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.1
  have c66150 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.1
  have c66151 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.1
  have c66152 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_14
  simp only [← c66147, ← c66148, ← c66149] at e_1070_14
  have e_1070_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_15
  simp only [← c66150, ← c66151, k2, ← c66152] at e_1070_15
  have hr := e_1070_15
  simp only [e_1070_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9226)) := by
  have c66153 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.1
  have c66154 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66155 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66156 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66157 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66158 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_16
  simp only [← c66153, ← c66154, ← c66155] at e_1070_16
  have e_1070_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_17
  simp only [← c66156, ← c66157, k2, ← c66158] at e_1070_17
  have hr := e_1070_17
  simp only [e_1070_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9227)) := by
  have c66159 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66160 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66161 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66162 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66163 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66164 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_18
  simp only [← c66159, ← c66160, ← c66161] at e_1070_18
  have e_1070_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_19
  simp only [← c66162, ← c66163, k2, ← c66164] at e_1070_19
  have hr := e_1070_19
  simp only [e_1070_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9228)) := by
  have c66165 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66166 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66167 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66168 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66169 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66170 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_0
  simp only [← c66165, ← c66166, ← c66167] at e_1071_0
  have e_1071_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_1
  simp only [← c66168, ← c66169, k2, ← c66170] at e_1071_1
  have hr := e_1071_1
  simp only [e_1071_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9229)) := by
  have c66171 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66172 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66173 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66174 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66175 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66176 := (publicBatchWrapper86_copies2068 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_2
  simp only [← c66171, ← c66172, ← c66173] at e_1071_2
  have e_1071_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_3
  simp only [← c66174, ← c66175, k2, ← c66176] at e_1071_3
  have hr := e_1071_3
  simp only [e_1071_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9230)) := by
  have c66177 := (publicBatchWrapper86_copies2068 a h).2.1
  have c66178 := (publicBatchWrapper86_copies2068 a h).2.2.1
  have c66179 := (publicBatchWrapper86_copies2068 a h).2.2.2.1
  have c66180 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.1
  have c66181 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.1
  have c66182 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_4
  simp only [← c66177, ← c66178, ← c66179] at e_1071_4
  have e_1071_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_5
  simp only [← c66180, ← c66181, k2, ← c66182] at e_1071_5
  have hr := e_1071_5
  simp only [e_1071_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9231)) := by
  have c66183 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.1
  have c66184 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.1
  have c66185 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.1
  have c66186 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66187 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66188 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_6
  simp only [← c66183, ← c66184, ← c66185] at e_1071_6
  have e_1071_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_7
  simp only [← c66186, ← c66187, k2, ← c66188] at e_1071_7
  have hr := e_1071_7
  simp only [e_1071_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9232)) := by
  have c66189 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66190 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66191 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66192 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66193 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66194 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_8
  simp only [← c66189, ← c66190, ← c66191] at e_1071_8
  have e_1071_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_9
  simp only [← c66192, ← c66193, k2, ← c66194] at e_1071_9
  have hr := e_1071_9
  simp only [e_1071_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9233)) := by
  have c66195 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66196 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66197 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66198 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66199 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66200 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_10
  simp only [← c66195, ← c66196, ← c66197] at e_1071_10
  have e_1071_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_11
  simp only [← c66198, ← c66199, k2, ← c66200] at e_1071_11
  have hr := e_1071_11
  simp only [e_1071_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9234)) := by
  have c66201 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66202 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66203 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66204 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66205 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66206 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_12
  simp only [← c66201, ← c66202, ← c66203] at e_1071_12
  have e_1071_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_13
  simp only [← c66204, ← c66205, k2, ← c66206] at e_1071_13
  have hr := e_1071_13
  simp only [e_1071_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9235)) := by
  have c66207 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66208 := (publicBatchWrapper86_copies2069 a h).1
  have c66209 := (publicBatchWrapper86_copies2069 a h).2.1
  have c66210 := (publicBatchWrapper86_copies2069 a h).2.2.1
  have c66211 := (publicBatchWrapper86_copies2069 a h).2.2.2.1
  have c66212 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_14
  simp only [← c66207, ← c66208, ← c66209] at e_1071_14
  have e_1071_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_15
  simp only [← c66210, ← c66211, k2, ← c66212] at e_1071_15
  have hr := e_1071_15
  simp only [e_1071_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9236)) := by
  have c66213 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.1
  have c66214 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.1
  have c66215 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.1
  have c66216 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.1
  have c66217 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.1
  have c66218 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_16
  simp only [← c66213, ← c66214, ← c66215] at e_1071_16
  have e_1071_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_17
  simp only [← c66216, ← c66217, k2, ← c66218] at e_1071_17
  have hr := e_1071_17
  simp only [e_1071_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9237)) := by
  have c66219 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66220 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66221 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66222 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66223 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66224 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_18
  simp only [← c66219, ← c66220, ← c66221] at e_1071_18
  have e_1071_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_19
  simp only [← c66222, ← c66223, k2, ← c66224] at e_1071_19
  have hr := e_1071_19
  simp only [e_1071_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9238)) := by
  have c66225 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66226 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66227 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66228 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66229 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66230 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_0
  simp only [← c66225, ← c66226, ← c66227] at e_1072_0
  have e_1072_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_1
  simp only [← c66228, ← c66229, k2, ← c66230] at e_1072_1
  have hr := e_1072_1
  simp only [e_1072_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9239)) := by
  have c66231 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66232 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66233 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66234 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66235 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66236 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_2
  simp only [← c66231, ← c66232, ← c66233] at e_1072_2
  have e_1072_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_3
  simp only [← c66234, ← c66235, k2, ← c66236] at e_1072_3
  have hr := e_1072_3
  simp only [e_1072_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9240)) := by
  have c66237 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66238 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66239 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66240 := (publicBatchWrapper86_copies2070 a h).1
  have c66241 := (publicBatchWrapper86_copies2070 a h).2.1
  have c66242 := (publicBatchWrapper86_copies2070 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_4
  simp only [← c66237, ← c66238, ← c66239] at e_1072_4
  have e_1072_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_5
  simp only [← c66240, ← c66241, k2, ← c66242] at e_1072_5
  have hr := e_1072_5
  simp only [e_1072_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9241)) := by
  have c66243 := (publicBatchWrapper86_copies2070 a h).2.2.2.1
  have c66244 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.1
  have c66245 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.1
  have c66246 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.1
  have c66247 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.1
  have c66248 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_6
  simp only [← c66243, ← c66244, ← c66245] at e_1072_6
  have e_1072_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_7
  simp only [← c66246, ← c66247, k2, ← c66248] at e_1072_7
  have hr := e_1072_7
  simp only [e_1072_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9242)) := by
  have c66249 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.1
  have c66250 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66251 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66252 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66253 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66254 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_8
  simp only [← c66249, ← c66250, ← c66251] at e_1072_8
  have e_1072_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_9
  simp only [← c66252, ← c66253, k2, ← c66254] at e_1072_9
  have hr := e_1072_9
  simp only [e_1072_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9243)) := by
  have c66255 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66256 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66257 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66258 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66259 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66260 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_10
  simp only [← c66255, ← c66256, ← c66257] at e_1072_10
  have e_1072_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_11
  simp only [← c66258, ← c66259, k2, ← c66260] at e_1072_11
  have hr := e_1072_11
  simp only [e_1072_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9244)) := by
  have c66261 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66262 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66263 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66264 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66265 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66266 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_12
  simp only [← c66261, ← c66262, ← c66263] at e_1072_12
  have e_1072_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_13
  simp only [← c66264, ← c66265, k2, ← c66266] at e_1072_13
  have hr := e_1072_13
  simp only [e_1072_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9245)) := by
  have c66267 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66268 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66269 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66270 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66271 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66272 := (publicBatchWrapper86_copies2071 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_14
  simp only [← c66267, ← c66268, ← c66269] at e_1072_14
  have e_1072_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_15
  simp only [← c66270, ← c66271, k2, ← c66272] at e_1072_15
  have hr := e_1072_15
  simp only [e_1072_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9246)) := by
  have c66273 := (publicBatchWrapper86_copies2071 a h).2.1
  have c66274 := (publicBatchWrapper86_copies2071 a h).2.2.1
  have c66275 := (publicBatchWrapper86_copies2071 a h).2.2.2.1
  have c66276 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.1
  have c66277 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.1
  have c66278 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_16
  simp only [← c66273, ← c66274, ← c66275] at e_1072_16
  have e_1072_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_17
  simp only [← c66276, ← c66277, k2, ← c66278] at e_1072_17
  have hr := e_1072_17
  simp only [e_1072_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9247)) := by
  have c66279 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.1
  have c66280 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.1
  have c66281 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.1
  have c66282 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66283 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66284 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_18
  simp only [← c66279, ← c66280, ← c66281] at e_1072_18
  have e_1072_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_19
  simp only [← c66282, ← c66283, k2, ← c66284] at e_1072_19
  have hr := e_1072_19
  simp only [e_1072_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9248)) := by
  have c66285 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66286 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66287 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66288 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66289 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66290 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_0
  simp only [← c66285, ← c66286, ← c66287] at e_1073_0
  have e_1073_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_1
  simp only [← c66288, ← c66289, k2, ← c66290] at e_1073_1
  have hr := e_1073_1
  simp only [e_1073_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9249)) := by
  have c66291 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66292 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66293 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66294 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66295 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66296 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_2
  simp only [← c66291, ← c66292, ← c66293] at e_1073_2
  have e_1073_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_3
  simp only [← c66294, ← c66295, k2, ← c66296] at e_1073_3
  have hr := e_1073_3
  simp only [e_1073_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9250)) := by
  have c66297 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66298 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66299 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66300 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66301 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66302 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_4
  simp only [← c66297, ← c66298, ← c66299] at e_1073_4
  have e_1073_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_5
  simp only [← c66300, ← c66301, k2, ← c66302] at e_1073_5
  have hr := e_1073_5
  simp only [e_1073_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9251)) := by
  have c66303 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66304 := (publicBatchWrapper86_copies2072 a h).1
  have c66305 := (publicBatchWrapper86_copies2072 a h).2.1
  have c66306 := (publicBatchWrapper86_copies2072 a h).2.2.1
  have c66307 := (publicBatchWrapper86_copies2072 a h).2.2.2.1
  have c66308 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_6
  simp only [← c66303, ← c66304, ← c66305] at e_1073_6
  have e_1073_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_7
  simp only [← c66306, ← c66307, k2, ← c66308] at e_1073_7
  have hr := e_1073_7
  simp only [e_1073_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9252)) := by
  have c66309 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.1
  have c66310 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.1
  have c66311 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.1
  have c66312 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.1
  have c66313 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.1
  have c66314 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_8
  simp only [← c66309, ← c66310, ← c66311] at e_1073_8
  have e_1073_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_9
  simp only [← c66312, ← c66313, k2, ← c66314] at e_1073_9
  have hr := e_1073_9
  simp only [e_1073_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9253)) := by
  have c66315 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66316 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66317 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66318 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66319 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66320 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_10
  simp only [← c66315, ← c66316, ← c66317] at e_1073_10
  have e_1073_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_11
  simp only [← c66318, ← c66319, k2, ← c66320] at e_1073_11
  have hr := e_1073_11
  simp only [e_1073_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9254)) := by
  have c66321 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66322 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66323 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66324 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66325 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66326 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_12
  simp only [← c66321, ← c66322, ← c66323] at e_1073_12
  have e_1073_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_13
  simp only [← c66324, ← c66325, k2, ← c66326] at e_1073_13
  have hr := e_1073_13
  simp only [e_1073_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9255)) := by
  have c66327 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66328 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66329 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66330 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66331 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66332 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_14
  simp only [← c66327, ← c66328, ← c66329] at e_1073_14
  have e_1073_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_15
  simp only [← c66330, ← c66331, k2, ← c66332] at e_1073_15
  have hr := e_1073_15
  simp only [e_1073_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9256)) := by
  have c66333 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66334 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66335 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66336 := (publicBatchWrapper86_copies2073 a h).1
  have c66337 := (publicBatchWrapper86_copies2073 a h).2.1
  have c66338 := (publicBatchWrapper86_copies2073 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_16
  simp only [← c66333, ← c66334, ← c66335] at e_1073_16
  have e_1073_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_17
  simp only [← c66336, ← c66337, k2, ← c66338] at e_1073_17
  have hr := e_1073_17
  simp only [e_1073_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9257)) := by
  have c66339 := (publicBatchWrapper86_copies2073 a h).2.2.2.1
  have c66340 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.1
  have c66341 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.1
  have c66342 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.1
  have c66343 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.1
  have c66344 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_18
  simp only [← c66339, ← c66340, ← c66341] at e_1073_18
  have e_1073_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_19
  simp only [← c66342, ← c66343, k2, ← c66344] at e_1073_19
  have hr := e_1073_19
  simp only [e_1073_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9258)) := by
  have c66345 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.1
  have c66346 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66347 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66348 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66349 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66350 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_0
  simp only [← c66345, ← c66346, ← c66347] at e_1074_0
  have e_1074_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_1
  simp only [← c66348, ← c66349, k2, ← c66350] at e_1074_1
  have hr := e_1074_1
  simp only [e_1074_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9259)) := by
  have c66351 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66352 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66353 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66354 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66355 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66356 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_2
  simp only [← c66351, ← c66352, ← c66353] at e_1074_2
  have e_1074_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_3
  simp only [← c66354, ← c66355, k2, ← c66356] at e_1074_3
  have hr := e_1074_3
  simp only [e_1074_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9260)) := by
  have c66357 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66358 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66359 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66360 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66361 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66362 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_4
  simp only [← c66357, ← c66358, ← c66359] at e_1074_4
  have e_1074_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_5
  simp only [← c66360, ← c66361, k2, ← c66362] at e_1074_5
  have hr := e_1074_5
  simp only [e_1074_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9261)) := by
  have c66363 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66364 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66365 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66366 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66367 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66368 := (publicBatchWrapper86_copies2074 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_6
  simp only [← c66363, ← c66364, ← c66365] at e_1074_6
  have e_1074_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_7
  simp only [← c66366, ← c66367, k2, ← c66368] at e_1074_7
  have hr := e_1074_7
  simp only [e_1074_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9262)) := by
  have c66369 := (publicBatchWrapper86_copies2074 a h).2.1
  have c66370 := (publicBatchWrapper86_copies2074 a h).2.2.1
  have c66371 := (publicBatchWrapper86_copies2074 a h).2.2.2.1
  have c66372 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.1
  have c66373 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.1
  have c66374 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_8
  simp only [← c66369, ← c66370, ← c66371] at e_1074_8
  have e_1074_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_9
  simp only [← c66372, ← c66373, k2, ← c66374] at e_1074_9
  have hr := e_1074_9
  simp only [e_1074_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9263)) := by
  have c66375 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.1
  have c66376 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.1
  have c66377 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.1
  have c66378 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66379 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66380 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_10
  simp only [← c66375, ← c66376, ← c66377] at e_1074_10
  have e_1074_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_11
  simp only [← c66378, ← c66379, k2, ← c66380] at e_1074_11
  have hr := e_1074_11
  simp only [e_1074_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9264)) := by
  have c66381 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66382 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66383 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66384 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66385 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66386 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_12
  simp only [← c66381, ← c66382, ← c66383] at e_1074_12
  have e_1074_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_13
  simp only [← c66384, ← c66385, k2, ← c66386] at e_1074_13
  have hr := e_1074_13
  simp only [e_1074_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9265)) := by
  have c66387 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66388 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66389 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66390 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66391 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66392 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_14
  simp only [← c66387, ← c66388, ← c66389] at e_1074_14
  have e_1074_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_15
  simp only [← c66390, ← c66391, k2, ← c66392] at e_1074_15
  have hr := e_1074_15
  simp only [e_1074_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9266)) := by
  have c66393 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66394 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66395 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66396 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66397 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66398 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_16
  simp only [← c66393, ← c66394, ← c66395] at e_1074_16
  have e_1074_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_17
  simp only [← c66396, ← c66397, k2, ← c66398] at e_1074_17
  have hr := e_1074_17
  simp only [e_1074_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9267)) := by
  have c66399 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66400 := (publicBatchWrapper86_copies2075 a h).1
  have c66401 := (publicBatchWrapper86_copies2075 a h).2.1
  have c66402 := (publicBatchWrapper86_copies2075 a h).2.2.1
  have c66403 := (publicBatchWrapper86_copies2075 a h).2.2.2.1
  have c66404 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_18
  simp only [← c66399, ← c66400, ← c66401] at e_1074_18
  have e_1074_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_19
  simp only [← c66402, ← c66403, k2, ← c66404] at e_1074_19
  have hr := e_1074_19
  simp only [e_1074_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9268)) := by
  have c66405 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.1
  have c66406 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.1
  have c66407 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.1
  have c66408 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.1
  have c66409 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.1
  have c66410 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_0
  simp only [← c66405, ← c66406, ← c66407] at e_1075_0
  have e_1075_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_1
  simp only [← c66408, ← c66409, k2, ← c66410] at e_1075_1
  have hr := e_1075_1
  simp only [e_1075_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9269)) := by
  have c66411 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66412 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66413 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66414 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66415 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66416 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_2
  simp only [← c66411, ← c66412, ← c66413] at e_1075_2
  have e_1075_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_3
  simp only [← c66414, ← c66415, k2, ← c66416] at e_1075_3
  have hr := e_1075_3
  simp only [e_1075_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9270)) := by
  have c66417 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66418 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66419 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66420 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66421 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66422 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_4
  simp only [← c66417, ← c66418, ← c66419] at e_1075_4
  have e_1075_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_5
  simp only [← c66420, ← c66421, k2, ← c66422] at e_1075_5
  have hr := e_1075_5
  simp only [e_1075_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9271)) := by
  have c66423 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66424 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66425 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66426 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66427 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66428 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_6
  simp only [← c66423, ← c66424, ← c66425] at e_1075_6
  have e_1075_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_7
  simp only [← c66426, ← c66427, k2, ← c66428] at e_1075_7
  have hr := e_1075_7
  simp only [e_1075_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9272)) := by
  have c66429 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66430 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66431 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66432 := (publicBatchWrapper86_copies2076 a h).1
  have c66433 := (publicBatchWrapper86_copies2076 a h).2.1
  have c66434 := (publicBatchWrapper86_copies2076 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_8
  simp only [← c66429, ← c66430, ← c66431] at e_1075_8
  have e_1075_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_9
  simp only [← c66432, ← c66433, k2, ← c66434] at e_1075_9
  have hr := e_1075_9
  simp only [e_1075_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9273)) := by
  have c66435 := (publicBatchWrapper86_copies2076 a h).2.2.2.1
  have c66436 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.1
  have c66437 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.1
  have c66438 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.1
  have c66439 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.1
  have c66440 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_10
  simp only [← c66435, ← c66436, ← c66437] at e_1075_10
  have e_1075_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_11
  simp only [← c66438, ← c66439, k2, ← c66440] at e_1075_11
  have hr := e_1075_11
  simp only [e_1075_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9274)) := by
  have c66441 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.1
  have c66442 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66443 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66444 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66445 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66446 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_12
  simp only [← c66441, ← c66442, ← c66443] at e_1075_12
  have e_1075_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_13
  simp only [← c66444, ← c66445, k2, ← c66446] at e_1075_13
  have hr := e_1075_13
  simp only [e_1075_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9275)) := by
  have c66447 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66448 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66449 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66450 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66451 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66452 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_14
  simp only [← c66447, ← c66448, ← c66449] at e_1075_14
  have e_1075_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_15
  simp only [← c66450, ← c66451, k2, ← c66452] at e_1075_15
  have hr := e_1075_15
  simp only [e_1075_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9276)) := by
  have c66453 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66454 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66455 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66456 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66457 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66458 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_16
  simp only [← c66453, ← c66454, ← c66455] at e_1075_16
  have e_1075_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_17
  simp only [← c66456, ← c66457, k2, ← c66458] at e_1075_17
  have hr := e_1075_17
  simp only [e_1075_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9277)) := by
  have c66459 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66460 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66461 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66462 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66463 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66464 := (publicBatchWrapper86_copies2077 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_18
  simp only [← c66459, ← c66460, ← c66461] at e_1075_18
  have e_1075_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_19
  simp only [← c66462, ← c66463, k2, ← c66464] at e_1075_19
  have hr := e_1075_19
  simp only [e_1075_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9278)) := by
  have c66465 := (publicBatchWrapper86_copies2077 a h).2.1
  have c66466 := (publicBatchWrapper86_copies2077 a h).2.2.1
  have c66467 := (publicBatchWrapper86_copies2077 a h).2.2.2.1
  have c66468 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.1
  have c66469 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.1
  have c66470 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_0
  simp only [← c66465, ← c66466, ← c66467] at e_1076_0
  have e_1076_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_1
  simp only [← c66468, ← c66469, k2, ← c66470] at e_1076_1
  have hr := e_1076_1
  simp only [e_1076_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9279)) := by
  have c66471 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.1
  have c66472 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.1
  have c66473 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.1
  have c66474 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66475 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66476 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_2
  simp only [← c66471, ← c66472, ← c66473] at e_1076_2
  have e_1076_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_3
  simp only [← c66474, ← c66475, k2, ← c66476] at e_1076_3
  have hr := e_1076_3
  simp only [e_1076_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9280)) := by
  have c66477 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66478 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66479 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66480 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66481 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66482 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_4
  simp only [← c66477, ← c66478, ← c66479] at e_1076_4
  have e_1076_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_5
  simp only [← c66480, ← c66481, k2, ← c66482] at e_1076_5
  have hr := e_1076_5
  simp only [e_1076_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9281)) := by
  have c66483 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66484 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66485 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66486 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66487 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66488 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_6
  simp only [← c66483, ← c66484, ← c66485] at e_1076_6
  have e_1076_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_7
  simp only [← c66486, ← c66487, k2, ← c66488] at e_1076_7
  have hr := e_1076_7
  simp only [e_1076_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9282)) := by
  have c66489 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66490 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66491 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66492 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66493 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66494 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_8
  simp only [← c66489, ← c66490, ← c66491] at e_1076_8
  have e_1076_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_9
  simp only [← c66492, ← c66493, k2, ← c66494] at e_1076_9
  have hr := e_1076_9
  simp only [e_1076_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9283)) := by
  have c66495 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66496 := (publicBatchWrapper86_copies2078 a h).1
  have c66497 := (publicBatchWrapper86_copies2078 a h).2.1
  have c66498 := (publicBatchWrapper86_copies2078 a h).2.2.1
  have c66499 := (publicBatchWrapper86_copies2078 a h).2.2.2.1
  have c66500 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_10
  simp only [← c66495, ← c66496, ← c66497] at e_1076_10
  have e_1076_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_11
  simp only [← c66498, ← c66499, k2, ← c66500] at e_1076_11
  have hr := e_1076_11
  simp only [e_1076_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9284)) := by
  have c66501 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.1
  have c66502 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.1
  have c66503 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.1
  have c66504 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.1
  have c66505 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.1
  have c66506 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_12
  simp only [← c66501, ← c66502, ← c66503] at e_1076_12
  have e_1076_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_13
  simp only [← c66504, ← c66505, k2, ← c66506] at e_1076_13
  have hr := e_1076_13
  simp only [e_1076_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9285)) := by
  have c66507 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66508 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66509 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66510 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66511 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66512 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_14
  simp only [← c66507, ← c66508, ← c66509] at e_1076_14
  have e_1076_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_15
  simp only [← c66510, ← c66511, k2, ← c66512] at e_1076_15
  have hr := e_1076_15
  simp only [e_1076_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9286)) := by
  have c66513 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66514 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66515 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66516 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66517 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66518 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_16
  simp only [← c66513, ← c66514, ← c66515] at e_1076_16
  have e_1076_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_17
  simp only [← c66516, ← c66517, k2, ← c66518] at e_1076_17
  have hr := e_1076_17
  simp only [e_1076_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9287)) := by
  have c66519 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66520 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66521 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66522 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66523 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66524 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_18
  simp only [← c66519, ← c66520, ← c66521] at e_1076_18
  have e_1076_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_19
  simp only [← c66522, ← c66523, k2, ← c66524] at e_1076_19
  have hr := e_1076_19
  simp only [e_1076_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9288)) := by
  have c66525 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66526 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66527 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66528 := (publicBatchWrapper86_copies2079 a h).1
  have c66529 := (publicBatchWrapper86_copies2079 a h).2.1
  have c66530 := (publicBatchWrapper86_copies2079 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_0
  simp only [← c66525, ← c66526, ← c66527] at e_1077_0
  have e_1077_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_1
  simp only [← c66528, ← c66529, k2, ← c66530] at e_1077_1
  have hr := e_1077_1
  simp only [e_1077_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9289)) := by
  have c66531 := (publicBatchWrapper86_copies2079 a h).2.2.2.1
  have c66532 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.1
  have c66533 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.1
  have c66534 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.1
  have c66535 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.1
  have c66536 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_2
  simp only [← c66531, ← c66532, ← c66533] at e_1077_2
  have e_1077_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_3
  simp only [← c66534, ← c66535, k2, ← c66536] at e_1077_3
  have hr := e_1077_3
  simp only [e_1077_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9290)) := by
  have c66537 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.1
  have c66538 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66539 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66540 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66541 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66542 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_4
  simp only [← c66537, ← c66538, ← c66539] at e_1077_4
  have e_1077_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_5
  simp only [← c66540, ← c66541, k2, ← c66542] at e_1077_5
  have hr := e_1077_5
  simp only [e_1077_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9291)) := by
  have c66543 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66544 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66545 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66546 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66547 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66548 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_6
  simp only [← c66543, ← c66544, ← c66545] at e_1077_6
  have e_1077_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_7
  simp only [← c66546, ← c66547, k2, ← c66548] at e_1077_7
  have hr := e_1077_7
  simp only [e_1077_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9292)) := by
  have c66549 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66550 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66551 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66552 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66553 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66554 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_8
  simp only [← c66549, ← c66550, ← c66551] at e_1077_8
  have e_1077_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_9
  simp only [← c66552, ← c66553, k2, ← c66554] at e_1077_9
  have hr := e_1077_9
  simp only [e_1077_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9293)) := by
  have c66555 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66556 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66557 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66558 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66559 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66560 := (publicBatchWrapper86_copies2080 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_10
  simp only [← c66555, ← c66556, ← c66557] at e_1077_10
  have e_1077_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_11
  simp only [← c66558, ← c66559, k2, ← c66560] at e_1077_11
  have hr := e_1077_11
  simp only [e_1077_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9294)) := by
  have c66561 := (publicBatchWrapper86_copies2080 a h).2.1
  have c66562 := (publicBatchWrapper86_copies2080 a h).2.2.1
  have c66563 := (publicBatchWrapper86_copies2080 a h).2.2.2.1
  have c66564 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.1
  have c66565 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.1
  have c66566 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_12
  simp only [← c66561, ← c66562, ← c66563] at e_1077_12
  have e_1077_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_13
  simp only [← c66564, ← c66565, k2, ← c66566] at e_1077_13
  have hr := e_1077_13
  simp only [e_1077_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9295)) := by
  have c66567 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.1
  have c66568 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.1
  have c66569 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.1
  have c66570 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66571 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66572 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_14
  simp only [← c66567, ← c66568, ← c66569] at e_1077_14
  have e_1077_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_15
  simp only [← c66570, ← c66571, k2, ← c66572] at e_1077_15
  have hr := e_1077_15
  simp only [e_1077_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9296)) := by
  have c66573 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66574 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66575 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66576 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66577 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66578 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_16
  simp only [← c66573, ← c66574, ← c66575] at e_1077_16
  have e_1077_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_17
  simp only [← c66576, ← c66577, k2, ← c66578] at e_1077_17
  have hr := e_1077_17
  simp only [e_1077_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9297)) := by
  have c66579 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66580 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66581 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66582 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66583 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66584 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_18
  simp only [← c66579, ← c66580, ← c66581] at e_1077_18
  have e_1077_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_19
  simp only [← c66582, ← c66583, k2, ← c66584] at e_1077_19
  have hr := e_1077_19
  simp only [e_1077_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9298)) := by
  have c66585 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66586 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66587 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66588 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66589 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66590 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_0
  simp only [← c66585, ← c66586, ← c66587] at e_1078_0
  have e_1078_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_1
  simp only [← c66588, ← c66589, k2, ← c66590] at e_1078_1
  have hr := e_1078_1
  simp only [e_1078_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9299)) := by
  have c66591 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66592 := (publicBatchWrapper86_copies2081 a h).1
  have c66593 := (publicBatchWrapper86_copies2081 a h).2.1
  have c66594 := (publicBatchWrapper86_copies2081 a h).2.2.1
  have c66595 := (publicBatchWrapper86_copies2081 a h).2.2.2.1
  have c66596 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_2
  simp only [← c66591, ← c66592, ← c66593] at e_1078_2
  have e_1078_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_3
  simp only [← c66594, ← c66595, k2, ← c66596] at e_1078_3
  have hr := e_1078_3
  simp only [e_1078_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9300)) := by
  have c66597 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.1
  have c66598 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.1
  have c66599 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.1
  have c66600 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.1
  have c66601 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.1
  have c66602 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_4
  simp only [← c66597, ← c66598, ← c66599] at e_1078_4
  have e_1078_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_5
  simp only [← c66600, ← c66601, k2, ← c66602] at e_1078_5
  have hr := e_1078_5
  simp only [e_1078_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9301)) := by
  have c66603 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66604 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66605 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66606 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66607 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66608 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_6
  simp only [← c66603, ← c66604, ← c66605] at e_1078_6
  have e_1078_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_7
  simp only [← c66606, ← c66607, k2, ← c66608] at e_1078_7
  have hr := e_1078_7
  simp only [e_1078_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9302)) := by
  have c66609 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66610 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66611 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66612 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66613 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66614 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_8
  simp only [← c66609, ← c66610, ← c66611] at e_1078_8
  have e_1078_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_9
  simp only [← c66612, ← c66613, k2, ← c66614] at e_1078_9
  have hr := e_1078_9
  simp only [e_1078_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9303)) := by
  have c66615 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66616 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66617 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66618 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66619 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66620 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_10
  simp only [← c66615, ← c66616, ← c66617] at e_1078_10
  have e_1078_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_11
  simp only [← c66618, ← c66619, k2, ← c66620] at e_1078_11
  have hr := e_1078_11
  simp only [e_1078_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9304)) := by
  have c66621 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66622 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66623 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66624 := (publicBatchWrapper86_copies2082 a h).1
  have c66625 := (publicBatchWrapper86_copies2082 a h).2.1
  have c66626 := (publicBatchWrapper86_copies2082 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_12
  simp only [← c66621, ← c66622, ← c66623] at e_1078_12
  have e_1078_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_13
  simp only [← c66624, ← c66625, k2, ← c66626] at e_1078_13
  have hr := e_1078_13
  simp only [e_1078_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9305)) := by
  have c66627 := (publicBatchWrapper86_copies2082 a h).2.2.2.1
  have c66628 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.1
  have c66629 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.1
  have c66630 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.1
  have c66631 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.1
  have c66632 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_14
  simp only [← c66627, ← c66628, ← c66629] at e_1078_14
  have e_1078_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_15
  simp only [← c66630, ← c66631, k2, ← c66632] at e_1078_15
  have hr := e_1078_15
  simp only [e_1078_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9306)) := by
  have c66633 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.1
  have c66634 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66635 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66636 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66637 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66638 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_16
  simp only [← c66633, ← c66634, ← c66635] at e_1078_16
  have e_1078_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_17
  simp only [← c66636, ← c66637, k2, ← c66638] at e_1078_17
  have hr := e_1078_17
  simp only [e_1078_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9307)) := by
  have c66639 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66640 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66641 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66642 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66643 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66644 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_18
  simp only [← c66639, ← c66640, ← c66641] at e_1078_18
  have e_1078_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_19
  simp only [← c66642, ← c66643, k2, ← c66644] at e_1078_19
  have hr := e_1078_19
  simp only [e_1078_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9308)) := by
  have c66645 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66646 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66647 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66648 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66649 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66650 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_0
  simp only [← c66645, ← c66646, ← c66647] at e_1079_0
  have e_1079_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_1
  simp only [← c66648, ← c66649, k2, ← c66650] at e_1079_1
  have hr := e_1079_1
  simp only [e_1079_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9309)) := by
  have c66651 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66652 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66653 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66654 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66655 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66656 := (publicBatchWrapper86_copies2083 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_2
  simp only [← c66651, ← c66652, ← c66653] at e_1079_2
  have e_1079_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_3
  simp only [← c66654, ← c66655, k2, ← c66656] at e_1079_3
  have hr := e_1079_3
  simp only [e_1079_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9310)) := by
  have c66657 := (publicBatchWrapper86_copies2083 a h).2.1
  have c66658 := (publicBatchWrapper86_copies2083 a h).2.2.1
  have c66659 := (publicBatchWrapper86_copies2083 a h).2.2.2.1
  have c66660 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.1
  have c66661 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.1
  have c66662 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_4
  simp only [← c66657, ← c66658, ← c66659] at e_1079_4
  have e_1079_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_5
  simp only [← c66660, ← c66661, k2, ← c66662] at e_1079_5
  have hr := e_1079_5
  simp only [e_1079_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9311)) := by
  have c66663 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.1
  have c66664 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.1
  have c66665 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.1
  have c66666 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66667 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66668 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_6
  simp only [← c66663, ← c66664, ← c66665] at e_1079_6
  have e_1079_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_7
  simp only [← c66666, ← c66667, k2, ← c66668] at e_1079_7
  have hr := e_1079_7
  simp only [e_1079_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9312)) := by
  have c66669 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66670 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66671 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66672 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66673 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66674 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_8
  simp only [← c66669, ← c66670, ← c66671] at e_1079_8
  have e_1079_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_9
  simp only [← c66672, ← c66673, k2, ← c66674] at e_1079_9
  have hr := e_1079_9
  simp only [e_1079_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9313)) := by
  have c66675 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66676 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66677 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66678 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66679 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66680 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_10
  simp only [← c66675, ← c66676, ← c66677] at e_1079_10
  have e_1079_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_11
  simp only [← c66678, ← c66679, k2, ← c66680] at e_1079_11
  have hr := e_1079_11
  simp only [e_1079_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9314)) := by
  have c66681 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66682 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66683 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66684 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66685 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66686 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_12
  simp only [← c66681, ← c66682, ← c66683] at e_1079_12
  have e_1079_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_13
  simp only [← c66684, ← c66685, k2, ← c66686] at e_1079_13
  have hr := e_1079_13
  simp only [e_1079_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9315)) := by
  have c66687 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66688 := (publicBatchWrapper86_copies2084 a h).1
  have c66689 := (publicBatchWrapper86_copies2084 a h).2.1
  have c66690 := (publicBatchWrapper86_copies2084 a h).2.2.1
  have c66691 := (publicBatchWrapper86_copies2084 a h).2.2.2.1
  have c66692 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_14
  simp only [← c66687, ← c66688, ← c66689] at e_1079_14
  have e_1079_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_15
  simp only [← c66690, ← c66691, k2, ← c66692] at e_1079_15
  have hr := e_1079_15
  simp only [e_1079_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9316)) := by
  have c66693 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.1
  have c66694 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.1
  have c66695 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.1
  have c66696 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.1
  have c66697 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.1
  have c66698 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_16
  simp only [← c66693, ← c66694, ← c66695] at e_1079_16
  have e_1079_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_17
  simp only [← c66696, ← c66697, k2, ← c66698] at e_1079_17
  have hr := e_1079_17
  simp only [e_1079_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9317)) := by
  have c66699 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66700 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66701 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66702 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66703 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66704 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_18
  simp only [← c66699, ← c66700, ← c66701] at e_1079_18
  have e_1079_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_19
  simp only [← c66702, ← c66703, k2, ← c66704] at e_1079_19
  have hr := e_1079_19
  simp only [e_1079_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9318)) := by
  have c66705 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66706 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66707 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66708 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66709 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66710 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_0
  simp only [← c66705, ← c66706, ← c66707] at e_1080_0
  have e_1080_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_1
  simp only [← c66708, ← c66709, k2, ← c66710] at e_1080_1
  have hr := e_1080_1
  simp only [e_1080_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9319)) := by
  have c66711 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66712 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66713 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66714 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66715 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66716 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_2
  simp only [← c66711, ← c66712, ← c66713] at e_1080_2
  have e_1080_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_3
  simp only [← c66714, ← c66715, k2, ← c66716] at e_1080_3
  have hr := e_1080_3
  simp only [e_1080_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9320)) := by
  have c66717 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66718 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66719 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66720 := (publicBatchWrapper86_copies2085 a h).1
  have c66721 := (publicBatchWrapper86_copies2085 a h).2.1
  have c66722 := (publicBatchWrapper86_copies2085 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_4
  simp only [← c66717, ← c66718, ← c66719] at e_1080_4
  have e_1080_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_5
  simp only [← c66720, ← c66721, k2, ← c66722] at e_1080_5
  have hr := e_1080_5
  simp only [e_1080_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9321)) := by
  have c66723 := (publicBatchWrapper86_copies2085 a h).2.2.2.1
  have c66724 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.1
  have c66725 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.1
  have c66726 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.1
  have c66727 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.1
  have c66728 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_6
  simp only [← c66723, ← c66724, ← c66725] at e_1080_6
  have e_1080_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_7
  simp only [← c66726, ← c66727, k2, ← c66728] at e_1080_7
  have hr := e_1080_7
  simp only [e_1080_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9322)) := by
  have c66729 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.1
  have c66730 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66731 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66732 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66733 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66734 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_8
  simp only [← c66729, ← c66730, ← c66731] at e_1080_8
  have e_1080_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_9
  simp only [← c66732, ← c66733, k2, ← c66734] at e_1080_9
  have hr := e_1080_9
  simp only [e_1080_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9323)) := by
  have c66735 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66736 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66737 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66738 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66739 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66740 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_10
  simp only [← c66735, ← c66736, ← c66737] at e_1080_10
  have e_1080_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_11
  simp only [← c66738, ← c66739, k2, ← c66740] at e_1080_11
  have hr := e_1080_11
  simp only [e_1080_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9324)) := by
  have c66741 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66742 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66743 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66744 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66745 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66746 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_12
  simp only [← c66741, ← c66742, ← c66743] at e_1080_12
  have e_1080_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_13
  simp only [← c66744, ← c66745, k2, ← c66746] at e_1080_13
  have hr := e_1080_13
  simp only [e_1080_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9325)) := by
  have c66747 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66748 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66749 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66750 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66751 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66752 := (publicBatchWrapper86_copies2086 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_14
  simp only [← c66747, ← c66748, ← c66749] at e_1080_14
  have e_1080_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_15
  simp only [← c66750, ← c66751, k2, ← c66752] at e_1080_15
  have hr := e_1080_15
  simp only [e_1080_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9326)) := by
  have c66753 := (publicBatchWrapper86_copies2086 a h).2.1
  have c66754 := (publicBatchWrapper86_copies2086 a h).2.2.1
  have c66755 := (publicBatchWrapper86_copies2086 a h).2.2.2.1
  have c66756 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.1
  have c66757 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.1
  have c66758 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_16
  simp only [← c66753, ← c66754, ← c66755] at e_1080_16
  have e_1080_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_17
  simp only [← c66756, ← c66757, k2, ← c66758] at e_1080_17
  have hr := e_1080_17
  simp only [e_1080_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9327)) := by
  have c66759 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.1
  have c66760 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.1
  have c66761 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.1
  have c66762 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66763 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66764 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_18
  simp only [← c66759, ← c66760, ← c66761] at e_1080_18
  have e_1080_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_19
  simp only [← c66762, ← c66763, k2, ← c66764] at e_1080_19
  have hr := e_1080_19
  simp only [e_1080_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9328)) := by
  have c66765 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66766 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66767 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66768 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66769 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66770 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_0
  simp only [← c66765, ← c66766, ← c66767] at e_1081_0
  have e_1081_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_1
  simp only [← c66768, ← c66769, k2, ← c66770] at e_1081_1
  have hr := e_1081_1
  simp only [e_1081_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9329)) := by
  have c66771 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66772 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66773 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66774 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66775 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66776 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_2
  simp only [← c66771, ← c66772, ← c66773] at e_1081_2
  have e_1081_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_3
  simp only [← c66774, ← c66775, k2, ← c66776] at e_1081_3
  have hr := e_1081_3
  simp only [e_1081_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9330)) := by
  have c66777 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66778 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66779 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66780 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66781 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66782 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_4
  simp only [← c66777, ← c66778, ← c66779] at e_1081_4
  have e_1081_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_5
  simp only [← c66780, ← c66781, k2, ← c66782] at e_1081_5
  have hr := e_1081_5
  simp only [e_1081_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9331)) := by
  have c66783 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66784 := (publicBatchWrapper86_copies2087 a h).1
  have c66785 := (publicBatchWrapper86_copies2087 a h).2.1
  have c66786 := (publicBatchWrapper86_copies2087 a h).2.2.1
  have c66787 := (publicBatchWrapper86_copies2087 a h).2.2.2.1
  have c66788 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_6
  simp only [← c66783, ← c66784, ← c66785] at e_1081_6
  have e_1081_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_7
  simp only [← c66786, ← c66787, k2, ← c66788] at e_1081_7
  have hr := e_1081_7
  simp only [e_1081_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9332)) := by
  have c66789 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.1
  have c66790 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.1
  have c66791 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.1
  have c66792 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.1
  have c66793 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.1
  have c66794 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_8
  simp only [← c66789, ← c66790, ← c66791] at e_1081_8
  have e_1081_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_9
  simp only [← c66792, ← c66793, k2, ← c66794] at e_1081_9
  have hr := e_1081_9
  simp only [e_1081_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9333)) := by
  have c66795 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66796 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66797 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66798 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66799 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66800 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_10
  simp only [← c66795, ← c66796, ← c66797] at e_1081_10
  have e_1081_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_11
  simp only [← c66798, ← c66799, k2, ← c66800] at e_1081_11
  have hr := e_1081_11
  simp only [e_1081_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9334)) := by
  have c66801 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66802 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66803 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66804 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66805 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66806 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_12
  simp only [← c66801, ← c66802, ← c66803] at e_1081_12
  have e_1081_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_13
  simp only [← c66804, ← c66805, k2, ← c66806] at e_1081_13
  have hr := e_1081_13
  simp only [e_1081_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9335)) := by
  have c66807 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66808 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66809 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66810 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66811 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66812 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_14
  simp only [← c66807, ← c66808, ← c66809] at e_1081_14
  have e_1081_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_15
  simp only [← c66810, ← c66811, k2, ← c66812] at e_1081_15
  have hr := e_1081_15
  simp only [e_1081_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9336)) := by
  have c66813 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66814 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66815 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66816 := (publicBatchWrapper86_copies2088 a h).1
  have c66817 := (publicBatchWrapper86_copies2088 a h).2.1
  have c66818 := (publicBatchWrapper86_copies2088 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_16
  simp only [← c66813, ← c66814, ← c66815] at e_1081_16
  have e_1081_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_17
  simp only [← c66816, ← c66817, k2, ← c66818] at e_1081_17
  have hr := e_1081_17
  simp only [e_1081_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9337)) := by
  have c66819 := (publicBatchWrapper86_copies2088 a h).2.2.2.1
  have c66820 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.1
  have c66821 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.1
  have c66822 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.1
  have c66823 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.1
  have c66824 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_18
  simp only [← c66819, ← c66820, ← c66821] at e_1081_18
  have e_1081_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_19
  simp only [← c66822, ← c66823, k2, ← c66824] at e_1081_19
  have hr := e_1081_19
  simp only [e_1081_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9338)) := by
  have c66825 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.1
  have c66826 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66827 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66828 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66829 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66830 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_0
  simp only [← c66825, ← c66826, ← c66827] at e_1082_0
  have e_1082_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_1
  simp only [← c66828, ← c66829, k2, ← c66830] at e_1082_1
  have hr := e_1082_1
  simp only [e_1082_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9339)) := by
  have c66831 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66832 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66833 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66834 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66835 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66836 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_2
  simp only [← c66831, ← c66832, ← c66833] at e_1082_2
  have e_1082_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_3
  simp only [← c66834, ← c66835, k2, ← c66836] at e_1082_3
  have hr := e_1082_3
  simp only [e_1082_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9340)) := by
  have c66837 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66838 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66839 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66840 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66841 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66842 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_4
  simp only [← c66837, ← c66838, ← c66839] at e_1082_4
  have e_1082_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_5
  simp only [← c66840, ← c66841, k2, ← c66842] at e_1082_5
  have hr := e_1082_5
  simp only [e_1082_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9341)) := by
  have c66843 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66844 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66845 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66846 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66847 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66848 := (publicBatchWrapper86_copies2089 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_6
  simp only [← c66843, ← c66844, ← c66845] at e_1082_6
  have e_1082_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_7
  simp only [← c66846, ← c66847, k2, ← c66848] at e_1082_7
  have hr := e_1082_7
  simp only [e_1082_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9342)) := by
  have c66849 := (publicBatchWrapper86_copies2089 a h).2.1
  have c66850 := (publicBatchWrapper86_copies2089 a h).2.2.1
  have c66851 := (publicBatchWrapper86_copies2089 a h).2.2.2.1
  have c66852 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.1
  have c66853 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.1
  have c66854 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_8
  simp only [← c66849, ← c66850, ← c66851] at e_1082_8
  have e_1082_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_9
  simp only [← c66852, ← c66853, k2, ← c66854] at e_1082_9
  have hr := e_1082_9
  simp only [e_1082_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9343)) := by
  have c66855 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.1
  have c66856 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.1
  have c66857 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.1
  have c66858 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66859 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66860 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_10
  simp only [← c66855, ← c66856, ← c66857] at e_1082_10
  have e_1082_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_11
  simp only [← c66858, ← c66859, k2, ← c66860] at e_1082_11
  have hr := e_1082_11
  simp only [e_1082_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9344)) := by
  have c66861 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66862 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66863 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66864 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66865 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66866 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_12
  simp only [← c66861, ← c66862, ← c66863] at e_1082_12
  have e_1082_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_13
  simp only [← c66864, ← c66865, k2, ← c66866] at e_1082_13
  have hr := e_1082_13
  simp only [e_1082_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9345)) := by
  have c66867 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66868 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66869 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66870 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66871 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66872 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_14
  simp only [← c66867, ← c66868, ← c66869] at e_1082_14
  have e_1082_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_15
  simp only [← c66870, ← c66871, k2, ← c66872] at e_1082_15
  have hr := e_1082_15
  simp only [e_1082_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9346)) := by
  have c66873 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66874 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66875 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66876 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66877 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66878 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_16
  simp only [← c66873, ← c66874, ← c66875] at e_1082_16
  have e_1082_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_17
  simp only [← c66876, ← c66877, k2, ← c66878] at e_1082_17
  have hr := e_1082_17
  simp only [e_1082_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9347)) := by
  have c66879 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66880 := (publicBatchWrapper86_copies2090 a h).1
  have c66881 := (publicBatchWrapper86_copies2090 a h).2.1
  have c66882 := (publicBatchWrapper86_copies2090 a h).2.2.1
  have c66883 := (publicBatchWrapper86_copies2090 a h).2.2.2.1
  have c66884 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_18
  simp only [← c66879, ← c66880, ← c66881] at e_1082_18
  have e_1082_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_19
  simp only [← c66882, ← c66883, k2, ← c66884] at e_1082_19
  have hr := e_1082_19
  simp only [e_1082_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9348)) := by
  have c66885 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.1
  have c66886 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.1
  have c66887 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.1
  have c66888 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.1
  have c66889 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.1
  have c66890 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_0
  simp only [← c66885, ← c66886, ← c66887] at e_1083_0
  have e_1083_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_1
  simp only [← c66888, ← c66889, k2, ← c66890] at e_1083_1
  have hr := e_1083_1
  simp only [e_1083_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9349)) := by
  have c66891 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66892 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66893 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66894 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66895 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66896 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_2
  simp only [← c66891, ← c66892, ← c66893] at e_1083_2
  have e_1083_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_3
  simp only [← c66894, ← c66895, k2, ← c66896] at e_1083_3
  have hr := e_1083_3
  simp only [e_1083_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9350)) := by
  have c66897 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66898 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66899 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66900 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66901 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66902 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_4
  simp only [← c66897, ← c66898, ← c66899] at e_1083_4
  have e_1083_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_5
  simp only [← c66900, ← c66901, k2, ← c66902] at e_1083_5
  have hr := e_1083_5
  simp only [e_1083_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9351)) := by
  have c66903 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66904 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66905 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66906 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66907 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66908 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_6
  simp only [← c66903, ← c66904, ← c66905] at e_1083_6
  have e_1083_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_7
  simp only [← c66906, ← c66907, k2, ← c66908] at e_1083_7
  have hr := e_1083_7
  simp only [e_1083_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9352)) := by
  have c66909 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66910 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66911 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66912 := (publicBatchWrapper86_copies2091 a h).1
  have c66913 := (publicBatchWrapper86_copies2091 a h).2.1
  have c66914 := (publicBatchWrapper86_copies2091 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_8
  simp only [← c66909, ← c66910, ← c66911] at e_1083_8
  have e_1083_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_9
  simp only [← c66912, ← c66913, k2, ← c66914] at e_1083_9
  have hr := e_1083_9
  simp only [e_1083_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9353)) := by
  have c66915 := (publicBatchWrapper86_copies2091 a h).2.2.2.1
  have c66916 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.1
  have c66917 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.1
  have c66918 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.1
  have c66919 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.1
  have c66920 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_10
  simp only [← c66915, ← c66916, ← c66917] at e_1083_10
  have e_1083_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_11
  simp only [← c66918, ← c66919, k2, ← c66920] at e_1083_11
  have hr := e_1083_11
  simp only [e_1083_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9354)) := by
  have c66921 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.1
  have c66922 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66923 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66924 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66925 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66926 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_12
  simp only [← c66921, ← c66922, ← c66923] at e_1083_12
  have e_1083_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_13
  simp only [← c66924, ← c66925, k2, ← c66926] at e_1083_13
  have hr := e_1083_13
  simp only [e_1083_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9355)) := by
  have c66927 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66928 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66929 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66930 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66931 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66932 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_14
  simp only [← c66927, ← c66928, ← c66929] at e_1083_14
  have e_1083_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_15
  simp only [← c66930, ← c66931, k2, ← c66932] at e_1083_15
  have hr := e_1083_15
  simp only [e_1083_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9356)) := by
  have c66933 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66934 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66935 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66936 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66937 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66938 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_16
  simp only [← c66933, ← c66934, ← c66935] at e_1083_16
  have e_1083_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_17
  simp only [← c66936, ← c66937, k2, ← c66938] at e_1083_17
  have hr := e_1083_17
  simp only [e_1083_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9357)) := by
  have c66939 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66940 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66941 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66942 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66943 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66944 := (publicBatchWrapper86_copies2092 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_18
  simp only [← c66939, ← c66940, ← c66941] at e_1083_18
  have e_1083_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_19
  simp only [← c66942, ← c66943, k2, ← c66944] at e_1083_19
  have hr := e_1083_19
  simp only [e_1083_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9358)) := by
  have c66945 := (publicBatchWrapper86_copies2092 a h).2.1
  have c66946 := (publicBatchWrapper86_copies2092 a h).2.2.1
  have c66947 := (publicBatchWrapper86_copies2092 a h).2.2.2.1
  have c66948 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.1
  have c66949 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.1
  have c66950 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_0
  simp only [← c66945, ← c66946, ← c66947] at e_1084_0
  have e_1084_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_1
  simp only [← c66948, ← c66949, k2, ← c66950] at e_1084_1
  have hr := e_1084_1
  simp only [e_1084_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9359)) := by
  have c66951 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.1
  have c66952 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.1
  have c66953 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.1
  have c66954 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66955 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66956 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_2
  simp only [← c66951, ← c66952, ← c66953] at e_1084_2
  have e_1084_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_3
  simp only [← c66954, ← c66955, k2, ← c66956] at e_1084_3
  have hr := e_1084_3
  simp only [e_1084_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9360)) := by
  have c66957 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66958 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66959 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66960 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66961 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66962 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_4
  simp only [← c66957, ← c66958, ← c66959] at e_1084_4
  have e_1084_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_5
  simp only [← c66960, ← c66961, k2, ← c66962] at e_1084_5
  have hr := e_1084_5
  simp only [e_1084_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9361)) := by
  have c66963 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66964 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66965 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66966 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66967 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66968 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_6
  simp only [← c66963, ← c66964, ← c66965] at e_1084_6
  have e_1084_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_7
  simp only [← c66966, ← c66967, k2, ← c66968] at e_1084_7
  have hr := e_1084_7
  simp only [e_1084_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9362)) := by
  have c66969 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66970 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66971 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66972 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66973 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66974 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_8
  simp only [← c66969, ← c66970, ← c66971] at e_1084_8
  have e_1084_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_9
  simp only [← c66972, ← c66973, k2, ← c66974] at e_1084_9
  have hr := e_1084_9
  simp only [e_1084_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9363)) := by
  have c66975 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66976 := (publicBatchWrapper86_copies2093 a h).1
  have c66977 := (publicBatchWrapper86_copies2093 a h).2.1
  have c66978 := (publicBatchWrapper86_copies2093 a h).2.2.1
  have c66979 := (publicBatchWrapper86_copies2093 a h).2.2.2.1
  have c66980 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_10
  simp only [← c66975, ← c66976, ← c66977] at e_1084_10
  have e_1084_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_11
  simp only [← c66978, ← c66979, k2, ← c66980] at e_1084_11
  have hr := e_1084_11
  simp only [e_1084_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9364)) := by
  have c66981 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.1
  have c66982 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.1
  have c66983 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.1
  have c66984 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.1
  have c66985 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.1
  have c66986 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_12
  simp only [← c66981, ← c66982, ← c66983] at e_1084_12
  have e_1084_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_13
  simp only [← c66984, ← c66985, k2, ← c66986] at e_1084_13
  have hr := e_1084_13
  simp only [e_1084_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9365)) := by
  have c66987 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66988 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66989 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66990 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66991 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66992 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_14
  simp only [← c66987, ← c66988, ← c66989] at e_1084_14
  have e_1084_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_15
  simp only [← c66990, ← c66991, k2, ← c66992] at e_1084_15
  have hr := e_1084_15
  simp only [e_1084_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9366)) := by
  have c66993 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66994 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66995 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66996 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66997 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66998 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_16
  simp only [← c66993, ← c66994, ← c66995] at e_1084_16
  have e_1084_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_17
  simp only [← c66996, ← c66997, k2, ← c66998] at e_1084_17
  have hr := e_1084_17
  simp only [e_1084_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9367)) := by
  have c66999 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67000 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67001 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67002 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67003 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67004 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_18
  simp only [← c66999, ← c67000, ← c67001] at e_1084_18
  have e_1084_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_19
  simp only [← c67002, ← c67003, k2, ← c67004] at e_1084_19
  have hr := e_1084_19
  simp only [e_1084_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9368)) := by
  have c67005 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67006 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67007 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67008 := (publicBatchWrapper86_copies2094 a h).1
  have c67009 := (publicBatchWrapper86_copies2094 a h).2.1
  have c67010 := (publicBatchWrapper86_copies2094 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_0
  simp only [← c67005, ← c67006, ← c67007] at e_1085_0
  have e_1085_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_1
  simp only [← c67008, ← c67009, k2, ← c67010] at e_1085_1
  have hr := e_1085_1
  simp only [e_1085_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9369)) := by
  have c67011 := (publicBatchWrapper86_copies2094 a h).2.2.2.1
  have c67012 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.1
  have c67013 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.1
  have c67014 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.1
  have c67015 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.1
  have c67016 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_2
  simp only [← c67011, ← c67012, ← c67013] at e_1085_2
  have e_1085_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_3
  simp only [← c67014, ← c67015, k2, ← c67016] at e_1085_3
  have hr := e_1085_3
  simp only [e_1085_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9410)) := by
  have c67017 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.1
  have c67018 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67019 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67020 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67021 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67022 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_4
  simp only [← c67017, ← c67018, ← c67019] at e_1085_4
  have e_1085_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_5
  simp only [← c67020, ← c67021, k2, ← c67022] at e_1085_5
  have hr := e_1085_5
  simp only [e_1085_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9411)) := by
  have c67023 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67024 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67025 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67026 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67027 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67028 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_6
  simp only [← c67023, ← c67024, ← c67025] at e_1085_6
  have e_1085_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_7
  simp only [← c67026, ← c67027, k2, ← c67028] at e_1085_7
  have hr := e_1085_7
  simp only [e_1085_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9412)) := by
  have c67029 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67030 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67031 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67032 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67033 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67034 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_8
  simp only [← c67029, ← c67030, ← c67031] at e_1085_8
  have e_1085_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_9
  simp only [← c67032, ← c67033, k2, ← c67034] at e_1085_9
  have hr := e_1085_9
  simp only [e_1085_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9413)) := by
  have c67035 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67036 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67037 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67038 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67039 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67040 := (publicBatchWrapper86_copies2095 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_10
  simp only [← c67035, ← c67036, ← c67037] at e_1085_10
  have e_1085_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_11
  simp only [← c67038, ← c67039, k2, ← c67040] at e_1085_11
  have hr := e_1085_11
  simp only [e_1085_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9414)) := by
  have c67041 := (publicBatchWrapper86_copies2095 a h).2.1
  have c67042 := (publicBatchWrapper86_copies2095 a h).2.2.1
  have c67043 := (publicBatchWrapper86_copies2095 a h).2.2.2.1
  have c67044 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.1
  have c67045 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.1
  have c67046 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_12
  simp only [← c67041, ← c67042, ← c67043] at e_1085_12
  have e_1085_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_13
  simp only [← c67044, ← c67045, k2, ← c67046] at e_1085_13
  have hr := e_1085_13
  simp only [e_1085_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9415)) := by
  have c67047 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.1
  have c67048 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.1
  have c67049 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.1
  have c67050 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67051 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67052 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_14
  simp only [← c67047, ← c67048, ← c67049] at e_1085_14
  have e_1085_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_15
  simp only [← c67050, ← c67051, k2, ← c67052] at e_1085_15
  have hr := e_1085_15
  simp only [e_1085_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9416)) := by
  have c67053 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67054 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67055 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67056 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67057 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67058 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_16
  simp only [← c67053, ← c67054, ← c67055] at e_1085_16
  have e_1085_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_17
  simp only [← c67056, ← c67057, k2, ← c67058] at e_1085_17
  have hr := e_1085_17
  simp only [e_1085_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9417)) := by
  have c67059 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67060 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67061 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67062 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67063 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67064 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_18
  simp only [← c67059, ← c67060, ← c67061] at e_1085_18
  have e_1085_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_19
  simp only [← c67062, ← c67063, k2, ← c67064] at e_1085_19
  have hr := e_1085_19
  simp only [e_1085_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9418)) := by
  have c67065 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67066 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67067 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67068 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67069 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67070 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_0
  simp only [← c67065, ← c67066, ← c67067] at e_1086_0
  have e_1086_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_1
  simp only [← c67068, ← c67069, k2, ← c67070] at e_1086_1
  have hr := e_1086_1
  simp only [e_1086_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9419)) := by
  have c67071 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67072 := (publicBatchWrapper86_copies2096 a h).1
  have c67073 := (publicBatchWrapper86_copies2096 a h).2.1
  have c67074 := (publicBatchWrapper86_copies2096 a h).2.2.1
  have c67075 := (publicBatchWrapper86_copies2096 a h).2.2.2.1
  have c67076 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_2
  simp only [← c67071, ← c67072, ← c67073] at e_1086_2
  have e_1086_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_3
  simp only [← c67074, ← c67075, k2, ← c67076] at e_1086_3
  have hr := e_1086_3
  simp only [e_1086_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9420)) := by
  have c67077 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.1
  have c67078 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.1
  have c67079 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.1
  have c67080 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.1
  have c67081 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.1
  have c67082 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_4
  simp only [← c67077, ← c67078, ← c67079] at e_1086_4
  have e_1086_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_5
  simp only [← c67080, ← c67081, k2, ← c67082] at e_1086_5
  have hr := e_1086_5
  simp only [e_1086_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9421)) := by
  have c67083 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67084 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67085 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67086 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67087 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67088 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_6
  simp only [← c67083, ← c67084, ← c67085] at e_1086_6
  have e_1086_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_7
  simp only [← c67086, ← c67087, k2, ← c67088] at e_1086_7
  have hr := e_1086_7
  simp only [e_1086_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9422)) := by
  have c67089 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67090 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67091 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67092 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67093 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67094 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_8
  simp only [← c67089, ← c67090, ← c67091] at e_1086_8
  have e_1086_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_9
  simp only [← c67092, ← c67093, k2, ← c67094] at e_1086_9
  have hr := e_1086_9
  simp only [e_1086_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9423)) := by
  have c67095 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67096 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67097 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67098 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67099 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67100 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_10
  simp only [← c67095, ← c67096, ← c67097] at e_1086_10
  have e_1086_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_11
  simp only [← c67098, ← c67099, k2, ← c67100] at e_1086_11
  have hr := e_1086_11
  simp only [e_1086_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9424)) := by
  have c67101 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67102 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67103 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67104 := (publicBatchWrapper86_copies2097 a h).1
  have c67105 := (publicBatchWrapper86_copies2097 a h).2.1
  have c67106 := (publicBatchWrapper86_copies2097 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_12
  simp only [← c67101, ← c67102, ← c67103] at e_1086_12
  have e_1086_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_13
  simp only [← c67104, ← c67105, k2, ← c67106] at e_1086_13
  have hr := e_1086_13
  simp only [e_1086_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9425)) := by
  have c67107 := (publicBatchWrapper86_copies2097 a h).2.2.2.1
  have c67108 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.1
  have c67109 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.1
  have c67110 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.1
  have c67111 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.1
  have c67112 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_14
  simp only [← c67107, ← c67108, ← c67109] at e_1086_14
  have e_1086_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_15
  simp only [← c67110, ← c67111, k2, ← c67112] at e_1086_15
  have hr := e_1086_15
  simp only [e_1086_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9426)) := by
  have c67113 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.1
  have c67114 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67115 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67116 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67117 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67118 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_16
  simp only [← c67113, ← c67114, ← c67115] at e_1086_16
  have e_1086_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_17
  simp only [← c67116, ← c67117, k2, ← c67118] at e_1086_17
  have hr := e_1086_17
  simp only [e_1086_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9427)) := by
  have c67119 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67120 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67121 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67122 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67123 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67124 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_18
  simp only [← c67119, ← c67120, ← c67121] at e_1086_18
  have e_1086_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_19
  simp only [← c67122, ← c67123, k2, ← c67124] at e_1086_19
  have hr := e_1086_19
  simp only [e_1086_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9428)) := by
  have c67125 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67126 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67127 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67128 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67129 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67130 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_0
  simp only [← c67125, ← c67126, ← c67127] at e_1087_0
  have e_1087_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_1
  simp only [← c67128, ← c67129, k2, ← c67130] at e_1087_1
  have hr := e_1087_1
  simp only [e_1087_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9429)) := by
  have c67131 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67132 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67133 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67134 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67135 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67136 := (publicBatchWrapper86_copies2098 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_2
  simp only [← c67131, ← c67132, ← c67133] at e_1087_2
  have e_1087_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_3
  simp only [← c67134, ← c67135, k2, ← c67136] at e_1087_3
  have hr := e_1087_3
  simp only [e_1087_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9430)) := by
  have c67137 := (publicBatchWrapper86_copies2098 a h).2.1
  have c67138 := (publicBatchWrapper86_copies2098 a h).2.2.1
  have c67139 := (publicBatchWrapper86_copies2098 a h).2.2.2.1
  have c67140 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.1
  have c67141 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.1
  have c67142 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_4
  simp only [← c67137, ← c67138, ← c67139] at e_1087_4
  have e_1087_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_5
  simp only [← c67140, ← c67141, k2, ← c67142] at e_1087_5
  have hr := e_1087_5
  simp only [e_1087_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9431)) := by
  have c67143 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.1
  have c67144 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.1
  have c67145 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.1
  have c67146 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67147 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67148 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_6
  simp only [← c67143, ← c67144, ← c67145] at e_1087_6
  have e_1087_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_7
  simp only [← c67146, ← c67147, k2, ← c67148] at e_1087_7
  have hr := e_1087_7
  simp only [e_1087_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9432)) := by
  have c67149 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67150 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67151 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67152 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67153 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67154 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_8
  simp only [← c67149, ← c67150, ← c67151] at e_1087_8
  have e_1087_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_9
  simp only [← c67152, ← c67153, k2, ← c67154] at e_1087_9
  have hr := e_1087_9
  simp only [e_1087_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9433)) := by
  have c67155 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67156 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67157 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67158 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67159 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67160 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_10
  simp only [← c67155, ← c67156, ← c67157] at e_1087_10
  have e_1087_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_11
  simp only [← c67158, ← c67159, k2, ← c67160] at e_1087_11
  have hr := e_1087_11
  simp only [e_1087_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9434)) := by
  have c67161 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67162 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67163 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67164 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67165 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67166 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_12
  simp only [← c67161, ← c67162, ← c67163] at e_1087_12
  have e_1087_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_13
  simp only [← c67164, ← c67165, k2, ← c67166] at e_1087_13
  have hr := e_1087_13
  simp only [e_1087_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9435)) := by
  have c67167 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67168 := (publicBatchWrapper86_copies2099 a h).1
  have c67169 := (publicBatchWrapper86_copies2099 a h).2.1
  have c67170 := (publicBatchWrapper86_copies2099 a h).2.2.1
  have c67171 := (publicBatchWrapper86_copies2099 a h).2.2.2.1
  have c67172 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_14
  simp only [← c67167, ← c67168, ← c67169] at e_1087_14
  have e_1087_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_15
  simp only [← c67170, ← c67171, k2, ← c67172] at e_1087_15
  have hr := e_1087_15
  simp only [e_1087_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9436)) := by
  have c67173 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.1
  have c67174 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.1
  have c67175 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.1
  have c67176 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.1
  have c67177 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.1
  have c67178 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_16
  simp only [← c67173, ← c67174, ← c67175] at e_1087_16
  have e_1087_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_17
  simp only [← c67176, ← c67177, k2, ← c67178] at e_1087_17
  have hr := e_1087_17
  simp only [e_1087_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9437)) := by
  have c67179 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67180 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67181 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67182 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67183 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67184 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_18
  simp only [← c67179, ← c67180, ← c67181] at e_1087_18
  have e_1087_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_19
  simp only [← c67182, ← c67183, k2, ← c67184] at e_1087_19
  have hr := e_1087_19
  simp only [e_1087_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9438)) := by
  have c67185 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67186 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67187 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67188 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67189 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67190 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_0
  simp only [← c67185, ← c67186, ← c67187] at e_1088_0
  have e_1088_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_1
  simp only [← c67188, ← c67189, k2, ← c67190] at e_1088_1
  have hr := e_1088_1
  simp only [e_1088_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9439)) := by
  have c67191 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67192 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67193 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67194 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67195 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67196 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_2
  simp only [← c67191, ← c67192, ← c67193] at e_1088_2
  have e_1088_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_3
  simp only [← c67194, ← c67195, k2, ← c67196] at e_1088_3
  have hr := e_1088_3
  simp only [e_1088_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9440)) := by
  have c67197 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67198 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67199 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67200 := (publicBatchWrapper86_copies2100 a h).1
  have c67201 := (publicBatchWrapper86_copies2100 a h).2.1
  have c67202 := (publicBatchWrapper86_copies2100 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_4
  simp only [← c67197, ← c67198, ← c67199] at e_1088_4
  have e_1088_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_5
  simp only [← c67200, ← c67201, k2, ← c67202] at e_1088_5
  have hr := e_1088_5
  simp only [e_1088_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9441)) := by
  have c67203 := (publicBatchWrapper86_copies2100 a h).2.2.2.1
  have c67204 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.1
  have c67205 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.1
  have c67206 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.1
  have c67207 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.1
  have c67208 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_6
  simp only [← c67203, ← c67204, ← c67205] at e_1088_6
  have e_1088_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_7
  simp only [← c67206, ← c67207, k2, ← c67208] at e_1088_7
  have hr := e_1088_7
  simp only [e_1088_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9442)) := by
  have c67209 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.1
  have c67210 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67211 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67212 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67213 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67214 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_8
  simp only [← c67209, ← c67210, ← c67211] at e_1088_8
  have e_1088_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_9
  simp only [← c67212, ← c67213, k2, ← c67214] at e_1088_9
  have hr := e_1088_9
  simp only [e_1088_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9443)) := by
  have c67215 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67216 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67217 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67218 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67219 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67220 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_10
  simp only [← c67215, ← c67216, ← c67217] at e_1088_10
  have e_1088_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_11
  simp only [← c67218, ← c67219, k2, ← c67220] at e_1088_11
  have hr := e_1088_11
  simp only [e_1088_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9444)) := by
  have c67221 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67222 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67223 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67224 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67225 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67226 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_12
  simp only [← c67221, ← c67222, ← c67223] at e_1088_12
  have e_1088_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_13
  simp only [← c67224, ← c67225, k2, ← c67226] at e_1088_13
  have hr := e_1088_13
  simp only [e_1088_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9445)) := by
  have c67227 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67228 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67229 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67230 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67231 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67232 := (publicBatchWrapper86_copies2101 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_14
  simp only [← c67227, ← c67228, ← c67229] at e_1088_14
  have e_1088_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_15
  simp only [← c67230, ← c67231, k2, ← c67232] at e_1088_15
  have hr := e_1088_15
  simp only [e_1088_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9446)) := by
  have c67233 := (publicBatchWrapper86_copies2101 a h).2.1
  have c67234 := (publicBatchWrapper86_copies2101 a h).2.2.1
  have c67235 := (publicBatchWrapper86_copies2101 a h).2.2.2.1
  have c67236 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.1
  have c67237 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.1
  have c67238 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_16
  simp only [← c67233, ← c67234, ← c67235] at e_1088_16
  have e_1088_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_17
  simp only [← c67236, ← c67237, k2, ← c67238] at e_1088_17
  have hr := e_1088_17
  simp only [e_1088_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9447)) := by
  have c67239 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.1
  have c67240 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.1
  have c67241 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.1
  have c67242 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67243 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67244 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_18
  simp only [← c67239, ← c67240, ← c67241] at e_1088_18
  have e_1088_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_19
  simp only [← c67242, ← c67243, k2, ← c67244] at e_1088_19
  have hr := e_1088_19
  simp only [e_1088_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9448)) := by
  have c67245 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67246 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67247 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67248 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67249 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67250 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_0
  simp only [← c67245, ← c67246, ← c67247] at e_1089_0
  have e_1089_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_1
  simp only [← c67248, ← c67249, k2, ← c67250] at e_1089_1
  have hr := e_1089_1
  simp only [e_1089_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9449)) := by
  have c67251 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67252 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67253 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67254 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67255 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67256 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_2
  simp only [← c67251, ← c67252, ← c67253] at e_1089_2
  have e_1089_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_3
  simp only [← c67254, ← c67255, k2, ← c67256] at e_1089_3
  have hr := e_1089_3
  simp only [e_1089_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9450)) := by
  have c67257 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67258 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67259 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67260 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67261 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67262 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_4
  simp only [← c67257, ← c67258, ← c67259] at e_1089_4
  have e_1089_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_5
  simp only [← c67260, ← c67261, k2, ← c67262] at e_1089_5
  have hr := e_1089_5
  simp only [e_1089_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9451)) := by
  have c67263 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67264 := (publicBatchWrapper86_copies2102 a h).1
  have c67265 := (publicBatchWrapper86_copies2102 a h).2.1
  have c67266 := (publicBatchWrapper86_copies2102 a h).2.2.1
  have c67267 := (publicBatchWrapper86_copies2102 a h).2.2.2.1
  have c67268 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_6
  simp only [← c67263, ← c67264, ← c67265] at e_1089_6
  have e_1089_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_7
  simp only [← c67266, ← c67267, k2, ← c67268] at e_1089_7
  have hr := e_1089_7
  simp only [e_1089_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9452)) := by
  have c67269 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.1
  have c67270 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.1
  have c67271 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.1
  have c67272 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.1
  have c67273 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.1
  have c67274 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_8
  simp only [← c67269, ← c67270, ← c67271] at e_1089_8
  have e_1089_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_9
  simp only [← c67272, ← c67273, k2, ← c67274] at e_1089_9
  have hr := e_1089_9
  simp only [e_1089_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9453)) := by
  have c67275 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67276 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67277 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67278 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67279 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67280 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_10
  simp only [← c67275, ← c67276, ← c67277] at e_1089_10
  have e_1089_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_11
  simp only [← c67278, ← c67279, k2, ← c67280] at e_1089_11
  have hr := e_1089_11
  simp only [e_1089_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9454)) := by
  have c67281 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67282 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67283 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67284 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67285 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67286 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_12
  simp only [← c67281, ← c67282, ← c67283] at e_1089_12
  have e_1089_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_13
  simp only [← c67284, ← c67285, k2, ← c67286] at e_1089_13
  have hr := e_1089_13
  simp only [e_1089_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9455)) := by
  have c67287 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67288 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67289 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67290 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67291 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67292 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_14
  simp only [← c67287, ← c67288, ← c67289] at e_1089_14
  have e_1089_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_15
  simp only [← c67290, ← c67291, k2, ← c67292] at e_1089_15
  have hr := e_1089_15
  simp only [e_1089_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9456)) := by
  have c67293 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67294 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67295 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67296 := (publicBatchWrapper86_copies2103 a h).1
  have c67297 := (publicBatchWrapper86_copies2103 a h).2.1
  have c67298 := (publicBatchWrapper86_copies2103 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_16
  simp only [← c67293, ← c67294, ← c67295] at e_1089_16
  have e_1089_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_17
  simp only [← c67296, ← c67297, k2, ← c67298] at e_1089_17
  have hr := e_1089_17
  simp only [e_1089_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9457)) := by
  have c67299 := (publicBatchWrapper86_copies2103 a h).2.2.2.1
  have c67300 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.1
  have c67301 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.1
  have c67302 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.1
  have c67303 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.1
  have c67304 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_18
  simp only [← c67299, ← c67300, ← c67301] at e_1089_18
  have e_1089_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_19
  simp only [← c67302, ← c67303, k2, ← c67304] at e_1089_19
  have hr := e_1089_19
  simp only [e_1089_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9458)) := by
  have c67305 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.1
  have c67306 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67307 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67308 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67309 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67310 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_0
  simp only [← c67305, ← c67306, ← c67307] at e_1090_0
  have e_1090_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_1
  simp only [← c67308, ← c67309, k2, ← c67310] at e_1090_1
  have hr := e_1090_1
  simp only [e_1090_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9459)) := by
  have c67311 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67312 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67313 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67314 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67315 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67316 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_2
  simp only [← c67311, ← c67312, ← c67313] at e_1090_2
  have e_1090_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_3
  simp only [← c67314, ← c67315, k2, ← c67316] at e_1090_3
  have hr := e_1090_3
  simp only [e_1090_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9460)) := by
  have c67317 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67318 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67319 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67320 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67321 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67322 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_4
  simp only [← c67317, ← c67318, ← c67319] at e_1090_4
  have e_1090_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_5
  simp only [← c67320, ← c67321, k2, ← c67322] at e_1090_5
  have hr := e_1090_5
  simp only [e_1090_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9461)) := by
  have c67323 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67324 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67325 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67326 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67327 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67328 := (publicBatchWrapper86_copies2104 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_6
  simp only [← c67323, ← c67324, ← c67325] at e_1090_6
  have e_1090_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_7
  simp only [← c67326, ← c67327, k2, ← c67328] at e_1090_7
  have hr := e_1090_7
  simp only [e_1090_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9462)) := by
  have c67329 := (publicBatchWrapper86_copies2104 a h).2.1
  have c67330 := (publicBatchWrapper86_copies2104 a h).2.2.1
  have c67331 := (publicBatchWrapper86_copies2104 a h).2.2.2.1
  have c67332 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.1
  have c67333 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.1
  have c67334 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_8
  simp only [← c67329, ← c67330, ← c67331] at e_1090_8
  have e_1090_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_9
  simp only [← c67332, ← c67333, k2, ← c67334] at e_1090_9
  have hr := e_1090_9
  simp only [e_1090_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9463)) := by
  have c67335 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.1
  have c67336 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.1
  have c67337 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.1
  have c67338 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67339 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67340 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_10
  simp only [← c67335, ← c67336, ← c67337] at e_1090_10
  have e_1090_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_11
  simp only [← c67338, ← c67339, k2, ← c67340] at e_1090_11
  have hr := e_1090_11
  simp only [e_1090_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9464)) := by
  have c67341 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67342 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67343 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67344 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67345 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67346 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_12
  simp only [← c67341, ← c67342, ← c67343] at e_1090_12
  have e_1090_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_13
  simp only [← c67344, ← c67345, k2, ← c67346] at e_1090_13
  have hr := e_1090_13
  simp only [e_1090_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9465)) := by
  have c67347 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67348 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67349 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67350 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67351 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67352 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_14
  simp only [← c67347, ← c67348, ← c67349] at e_1090_14
  have e_1090_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_15
  simp only [← c67350, ← c67351, k2, ← c67352] at e_1090_15
  have hr := e_1090_15
  simp only [e_1090_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9466)) := by
  have c67353 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67354 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67355 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67356 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67357 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67358 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_16
  simp only [← c67353, ← c67354, ← c67355] at e_1090_16
  have e_1090_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_17
  simp only [← c67356, ← c67357, k2, ← c67358] at e_1090_17
  have hr := e_1090_17
  simp only [e_1090_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9467)) := by
  have c67359 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67360 := (publicBatchWrapper86_copies2105 a h).1
  have c67361 := (publicBatchWrapper86_copies2105 a h).2.1
  have c67362 := (publicBatchWrapper86_copies2105 a h).2.2.1
  have c67363 := (publicBatchWrapper86_copies2105 a h).2.2.2.1
  have c67364 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_18
  simp only [← c67359, ← c67360, ← c67361] at e_1090_18
  have e_1090_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_19
  simp only [← c67362, ← c67363, k2, ← c67364] at e_1090_19
  have hr := e_1090_19
  simp only [e_1090_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9468)) := by
  have c67365 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.1
  have c67366 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.1
  have c67367 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.1
  have c67368 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.1
  have c67369 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.1
  have c67370 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_0
  simp only [← c67365, ← c67366, ← c67367] at e_1091_0
  have e_1091_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_1
  simp only [← c67368, ← c67369, k2, ← c67370] at e_1091_1
  have hr := e_1091_1
  simp only [e_1091_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9469)) := by
  have c67371 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67372 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67373 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67374 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67375 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67376 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_2
  simp only [← c67371, ← c67372, ← c67373] at e_1091_2
  have e_1091_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_3
  simp only [← c67374, ← c67375, k2, ← c67376] at e_1091_3
  have hr := e_1091_3
  simp only [e_1091_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9470)) := by
  have c67377 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67378 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67379 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67380 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67381 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67382 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_4
  simp only [← c67377, ← c67378, ← c67379] at e_1091_4
  have e_1091_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_5
  simp only [← c67380, ← c67381, k2, ← c67382] at e_1091_5
  have hr := e_1091_5
  simp only [e_1091_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9471)) := by
  have c67383 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67384 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67385 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67386 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67387 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67388 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_6
  simp only [← c67383, ← c67384, ← c67385] at e_1091_6
  have e_1091_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_7
  simp only [← c67386, ← c67387, k2, ← c67388] at e_1091_7
  have hr := e_1091_7
  simp only [e_1091_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9472)) := by
  have c67389 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67390 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67391 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67392 := (publicBatchWrapper86_copies2106 a h).1
  have c67393 := (publicBatchWrapper86_copies2106 a h).2.1
  have c67394 := (publicBatchWrapper86_copies2106 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_8
  simp only [← c67389, ← c67390, ← c67391] at e_1091_8
  have e_1091_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_9
  simp only [← c67392, ← c67393, k2, ← c67394] at e_1091_9
  have hr := e_1091_9
  simp only [e_1091_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9473)) := by
  have c67395 := (publicBatchWrapper86_copies2106 a h).2.2.2.1
  have c67396 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.1
  have c67397 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.1
  have c67398 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.1
  have c67399 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.1
  have c67400 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_10
  simp only [← c67395, ← c67396, ← c67397] at e_1091_10
  have e_1091_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_11
  simp only [← c67398, ← c67399, k2, ← c67400] at e_1091_11
  have hr := e_1091_11
  simp only [e_1091_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9474)) := by
  have c67401 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.1
  have c67402 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67403 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67404 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67405 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67406 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_12
  simp only [← c67401, ← c67402, ← c67403] at e_1091_12
  have e_1091_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_13
  simp only [← c67404, ← c67405, k2, ← c67406] at e_1091_13
  have hr := e_1091_13
  simp only [e_1091_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9475)) := by
  have c67407 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67408 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67409 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67410 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67411 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67412 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_14
  simp only [← c67407, ← c67408, ← c67409] at e_1091_14
  have e_1091_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_15
  simp only [← c67410, ← c67411, k2, ← c67412] at e_1091_15
  have hr := e_1091_15
  simp only [e_1091_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9476)) := by
  have c67413 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67414 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67415 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67416 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67417 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67418 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_16
  simp only [← c67413, ← c67414, ← c67415] at e_1091_16
  have e_1091_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_17
  simp only [← c67416, ← c67417, k2, ← c67418] at e_1091_17
  have hr := e_1091_17
  simp only [e_1091_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9477)) := by
  have c67419 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67420 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67421 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67422 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67423 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67424 := (publicBatchWrapper86_copies2107 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_18
  simp only [← c67419, ← c67420, ← c67421] at e_1091_18
  have e_1091_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_19
  simp only [← c67422, ← c67423, k2, ← c67424] at e_1091_19
  have hr := e_1091_19
  simp only [e_1091_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9478)) := by
  have c67425 := (publicBatchWrapper86_copies2107 a h).2.1
  have c67426 := (publicBatchWrapper86_copies2107 a h).2.2.1
  have c67427 := (publicBatchWrapper86_copies2107 a h).2.2.2.1
  have c67428 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.1
  have c67429 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.1
  have c67430 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_0
  simp only [← c67425, ← c67426, ← c67427] at e_1092_0
  have e_1092_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_1
  simp only [← c67428, ← c67429, k2, ← c67430] at e_1092_1
  have hr := e_1092_1
  simp only [e_1092_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9479)) := by
  have c67431 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.1
  have c67432 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.1
  have c67433 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.1
  have c67434 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67435 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67436 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_2
  simp only [← c67431, ← c67432, ← c67433] at e_1092_2
  have e_1092_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_3
  simp only [← c67434, ← c67435, k2, ← c67436] at e_1092_3
  have hr := e_1092_3
  simp only [e_1092_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9480)) := by
  have c67437 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67438 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67439 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67440 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67441 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67442 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_4
  simp only [← c67437, ← c67438, ← c67439] at e_1092_4
  have e_1092_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_5
  simp only [← c67440, ← c67441, k2, ← c67442] at e_1092_5
  have hr := e_1092_5
  simp only [e_1092_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9481)) := by
  have c67443 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67444 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67445 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67446 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67447 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67448 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_6
  simp only [← c67443, ← c67444, ← c67445] at e_1092_6
  have e_1092_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_7
  simp only [← c67446, ← c67447, k2, ← c67448] at e_1092_7
  have hr := e_1092_7
  simp only [e_1092_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9482)) := by
  have c67449 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67450 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67451 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67452 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67453 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67454 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_8
  simp only [← c67449, ← c67450, ← c67451] at e_1092_8
  have e_1092_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_9
  simp only [← c67452, ← c67453, k2, ← c67454] at e_1092_9
  have hr := e_1092_9
  simp only [e_1092_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9483)) := by
  have c67455 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67456 := (publicBatchWrapper86_copies2108 a h).1
  have c67457 := (publicBatchWrapper86_copies2108 a h).2.1
  have c67458 := (publicBatchWrapper86_copies2108 a h).2.2.1
  have c67459 := (publicBatchWrapper86_copies2108 a h).2.2.2.1
  have c67460 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_10
  simp only [← c67455, ← c67456, ← c67457] at e_1092_10
  have e_1092_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_11
  simp only [← c67458, ← c67459, k2, ← c67460] at e_1092_11
  have hr := e_1092_11
  simp only [e_1092_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9484)) := by
  have c67461 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.1
  have c67462 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.1
  have c67463 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.1
  have c67464 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.1
  have c67465 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.1
  have c67466 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_12
  simp only [← c67461, ← c67462, ← c67463] at e_1092_12
  have e_1092_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_13
  simp only [← c67464, ← c67465, k2, ← c67466] at e_1092_13
  have hr := e_1092_13
  simp only [e_1092_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9485)) := by
  have c67467 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67468 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67469 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67470 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67471 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67472 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_14
  simp only [← c67467, ← c67468, ← c67469] at e_1092_14
  have e_1092_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_15
  simp only [← c67470, ← c67471, k2, ← c67472] at e_1092_15
  have hr := e_1092_15
  simp only [e_1092_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9486)) := by
  have c67473 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67474 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67475 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67476 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67477 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67478 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_16
  simp only [← c67473, ← c67474, ← c67475] at e_1092_16
  have e_1092_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_17
  simp only [← c67476, ← c67477, k2, ← c67478] at e_1092_17
  have hr := e_1092_17
  simp only [e_1092_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9487)) := by
  have c67479 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67480 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67481 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67482 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67483 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67484 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_18
  simp only [← c67479, ← c67480, ← c67481] at e_1092_18
  have e_1092_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_19
  simp only [← c67482, ← c67483, k2, ← c67484] at e_1092_19
  have hr := e_1092_19
  simp only [e_1092_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9488)) := by
  have c67485 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67486 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67487 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67488 := (publicBatchWrapper86_copies2109 a h).1
  have c67489 := (publicBatchWrapper86_copies2109 a h).2.1
  have c67490 := (publicBatchWrapper86_copies2109 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_0
  simp only [← c67485, ← c67486, ← c67487] at e_1093_0
  have e_1093_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_1
  simp only [← c67488, ← c67489, k2, ← c67490] at e_1093_1
  have hr := e_1093_1
  simp only [e_1093_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9489)) := by
  have c67491 := (publicBatchWrapper86_copies2109 a h).2.2.2.1
  have c67492 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.1
  have c67493 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.1
  have c67494 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.1
  have c67495 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.1
  have c67496 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_2
  simp only [← c67491, ← c67492, ← c67493] at e_1093_2
  have e_1093_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_3
  simp only [← c67494, ← c67495, k2, ← c67496] at e_1093_3
  have hr := e_1093_3
  simp only [e_1093_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9490)) := by
  have c67497 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.1
  have c67498 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67499 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67500 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67501 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67502 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_4
  simp only [← c67497, ← c67498, ← c67499] at e_1093_4
  have e_1093_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_5
  simp only [← c67500, ← c67501, k2, ← c67502] at e_1093_5
  have hr := e_1093_5
  simp only [e_1093_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9491)) := by
  have c67503 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67504 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67505 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67506 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67507 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67508 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_6
  simp only [← c67503, ← c67504, ← c67505] at e_1093_6
  have e_1093_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_7
  simp only [← c67506, ← c67507, k2, ← c67508] at e_1093_7
  have hr := e_1093_7
  simp only [e_1093_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9492)) := by
  have c67509 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67510 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67511 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67512 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67513 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67514 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_8
  simp only [← c67509, ← c67510, ← c67511] at e_1093_8
  have e_1093_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_9
  simp only [← c67512, ← c67513, k2, ← c67514] at e_1093_9
  have hr := e_1093_9
  simp only [e_1093_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9493)) := by
  have c67515 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67516 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67517 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67518 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67519 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67520 := (publicBatchWrapper86_copies2110 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_10
  simp only [← c67515, ← c67516, ← c67517] at e_1093_10
  have e_1093_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_11
  simp only [← c67518, ← c67519, k2, ← c67520] at e_1093_11
  have hr := e_1093_11
  simp only [e_1093_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9494)) := by
  have c67521 := (publicBatchWrapper86_copies2110 a h).2.1
  have c67522 := (publicBatchWrapper86_copies2110 a h).2.2.1
  have c67523 := (publicBatchWrapper86_copies2110 a h).2.2.2.1
  have c67524 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.1
  have c67525 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.1
  have c67526 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_12
  simp only [← c67521, ← c67522, ← c67523] at e_1093_12
  have e_1093_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_13
  simp only [← c67524, ← c67525, k2, ← c67526] at e_1093_13
  have hr := e_1093_13
  simp only [e_1093_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9495)) := by
  have c67527 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.1
  have c67528 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.1
  have c67529 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.1
  have c67530 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67531 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67532 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_14
  simp only [← c67527, ← c67528, ← c67529] at e_1093_14
  have e_1093_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_15
  simp only [← c67530, ← c67531, k2, ← c67532] at e_1093_15
  have hr := e_1093_15
  simp only [e_1093_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9496)) := by
  have c67533 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67534 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67535 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67536 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67537 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67538 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_16
  simp only [← c67533, ← c67534, ← c67535] at e_1093_16
  have e_1093_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_17
  simp only [← c67536, ← c67537, k2, ← c67538] at e_1093_17
  have hr := e_1093_17
  simp only [e_1093_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9497)) := by
  have c67539 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67540 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67541 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67542 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67543 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67544 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_18
  simp only [← c67539, ← c67540, ← c67541] at e_1093_18
  have e_1093_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_19
  simp only [← c67542, ← c67543, k2, ← c67544] at e_1093_19
  have hr := e_1093_19
  simp only [e_1093_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9498)) := by
  have c67545 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67546 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67547 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67548 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67549 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67550 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_0
  simp only [← c67545, ← c67546, ← c67547] at e_1094_0
  have e_1094_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_1
  simp only [← c67548, ← c67549, k2, ← c67550] at e_1094_1
  have hr := e_1094_1
  simp only [e_1094_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9499)) := by
  have c67551 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67552 := (publicBatchWrapper86_copies2111 a h).1
  have c67553 := (publicBatchWrapper86_copies2111 a h).2.1
  have c67554 := (publicBatchWrapper86_copies2111 a h).2.2.1
  have c67555 := (publicBatchWrapper86_copies2111 a h).2.2.2.1
  have c67556 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_2
  simp only [← c67551, ← c67552, ← c67553] at e_1094_2
  have e_1094_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_3
  simp only [← c67554, ← c67555, k2, ← c67556] at e_1094_3
  have hr := e_1094_3
  simp only [e_1094_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9500)) := by
  have c67557 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.1
  have c67558 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.1
  have c67559 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.1
  have c67560 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.1
  have c67561 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.1
  have c67562 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_4
  simp only [← c67557, ← c67558, ← c67559] at e_1094_4
  have e_1094_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_5
  simp only [← c67560, ← c67561, k2, ← c67562] at e_1094_5
  have hr := e_1094_5
  simp only [e_1094_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9501)) := by
  have c67563 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67564 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67565 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67566 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67567 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67568 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_6
  simp only [← c67563, ← c67564, ← c67565] at e_1094_6
  have e_1094_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_7
  simp only [← c67566, ← c67567, k2, ← c67568] at e_1094_7
  have hr := e_1094_7
  simp only [e_1094_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9502)) := by
  have c67569 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67570 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67571 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67572 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67573 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67574 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_8
  simp only [← c67569, ← c67570, ← c67571] at e_1094_8
  have e_1094_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_9
  simp only [← c67572, ← c67573, k2, ← c67574] at e_1094_9
  have hr := e_1094_9
  simp only [e_1094_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9503)) := by
  have c67575 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67576 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67577 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67578 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67579 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67580 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_10
  simp only [← c67575, ← c67576, ← c67577] at e_1094_10
  have e_1094_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_11
  simp only [← c67578, ← c67579, k2, ← c67580] at e_1094_11
  have hr := e_1094_11
  simp only [e_1094_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9504)) := by
  have c67581 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67582 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67583 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67584 := (publicBatchWrapper86_copies2112 a h).1
  have c67585 := (publicBatchWrapper86_copies2112 a h).2.1
  have c67586 := (publicBatchWrapper86_copies2112 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_12
  simp only [← c67581, ← c67582, ← c67583] at e_1094_12
  have e_1094_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_13
  simp only [← c67584, ← c67585, k2, ← c67586] at e_1094_13
  have hr := e_1094_13
  simp only [e_1094_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9505)) := by
  have c67587 := (publicBatchWrapper86_copies2112 a h).2.2.2.1
  have c67588 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.1
  have c67589 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.1
  have c67590 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.1
  have c67591 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.1
  have c67592 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_14
  simp only [← c67587, ← c67588, ← c67589] at e_1094_14
  have e_1094_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_15
  simp only [← c67590, ← c67591, k2, ← c67592] at e_1094_15
  have hr := e_1094_15
  simp only [e_1094_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9506)) := by
  have c67593 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.1
  have c67594 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67595 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67596 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67597 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67598 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_16
  simp only [← c67593, ← c67594, ← c67595] at e_1094_16
  have e_1094_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_17
  simp only [← c67596, ← c67597, k2, ← c67598] at e_1094_17
  have hr := e_1094_17
  simp only [e_1094_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9507)) := by
  have c67599 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67600 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67601 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67602 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67603 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67604 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_18
  simp only [← c67599, ← c67600, ← c67601] at e_1094_18
  have e_1094_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_19
  simp only [← c67602, ← c67603, k2, ← c67604] at e_1094_19
  have hr := e_1094_19
  simp only [e_1094_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9508)) := by
  have c67605 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67606 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67607 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67608 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67609 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67610 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_0
  simp only [← c67605, ← c67606, ← c67607] at e_1095_0
  have e_1095_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_1
  simp only [← c67608, ← c67609, k2, ← c67610] at e_1095_1
  have hr := e_1095_1
  simp only [e_1095_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9509)) := by
  have c67611 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67612 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67613 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67614 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67615 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67616 := (publicBatchWrapper86_copies2113 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_2
  simp only [← c67611, ← c67612, ← c67613] at e_1095_2
  have e_1095_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_3
  simp only [← c67614, ← c67615, k2, ← c67616] at e_1095_3
  have hr := e_1095_3
  simp only [e_1095_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9510)) := by
  have c67617 := (publicBatchWrapper86_copies2113 a h).2.1
  have c67618 := (publicBatchWrapper86_copies2113 a h).2.2.1
  have c67619 := (publicBatchWrapper86_copies2113 a h).2.2.2.1
  have c67620 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.1
  have c67621 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.1
  have c67622 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_4
  simp only [← c67617, ← c67618, ← c67619] at e_1095_4
  have e_1095_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_5
  simp only [← c67620, ← c67621, k2, ← c67622] at e_1095_5
  have hr := e_1095_5
  simp only [e_1095_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9511)) := by
  have c67623 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.1
  have c67624 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.1
  have c67625 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.1
  have c67626 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67627 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67628 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_6
  simp only [← c67623, ← c67624, ← c67625] at e_1095_6
  have e_1095_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_7
  simp only [← c67626, ← c67627, k2, ← c67628] at e_1095_7
  have hr := e_1095_7
  simp only [e_1095_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9512)) := by
  have c67629 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67630 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67631 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67632 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67633 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67634 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_8
  simp only [← c67629, ← c67630, ← c67631] at e_1095_8
  have e_1095_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_9
  simp only [← c67632, ← c67633, k2, ← c67634] at e_1095_9
  have hr := e_1095_9
  simp only [e_1095_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9513)) := by
  have c67635 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67636 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67637 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67638 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67639 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67640 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_10
  simp only [← c67635, ← c67636, ← c67637] at e_1095_10
  have e_1095_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_11
  simp only [← c67638, ← c67639, k2, ← c67640] at e_1095_11
  have hr := e_1095_11
  simp only [e_1095_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9514)) := by
  have c67641 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67642 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67643 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67644 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67645 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67646 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_12
  simp only [← c67641, ← c67642, ← c67643] at e_1095_12
  have e_1095_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_13
  simp only [← c67644, ← c67645, k2, ← c67646] at e_1095_13
  have hr := e_1095_13
  simp only [e_1095_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9515)) := by
  have c67647 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67648 := (publicBatchWrapper86_copies2114 a h).1
  have c67649 := (publicBatchWrapper86_copies2114 a h).2.1
  have c67650 := (publicBatchWrapper86_copies2114 a h).2.2.1
  have c67651 := (publicBatchWrapper86_copies2114 a h).2.2.2.1
  have c67652 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_14
  simp only [← c67647, ← c67648, ← c67649] at e_1095_14
  have e_1095_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_15
  simp only [← c67650, ← c67651, k2, ← c67652] at e_1095_15
  have hr := e_1095_15
  simp only [e_1095_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9516)) := by
  have c67653 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.1
  have c67654 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.1
  have c67655 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.1
  have c67656 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.1
  have c67657 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.1
  have c67658 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_16
  simp only [← c67653, ← c67654, ← c67655] at e_1095_16
  have e_1095_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_17
  simp only [← c67656, ← c67657, k2, ← c67658] at e_1095_17
  have hr := e_1095_17
  simp only [e_1095_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9517)) := by
  have c67659 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67660 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67661 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67662 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67663 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67664 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_18
  simp only [← c67659, ← c67660, ← c67661] at e_1095_18
  have e_1095_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_19
  simp only [← c67662, ← c67663, k2, ← c67664] at e_1095_19
  have hr := e_1095_19
  simp only [e_1095_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9518)) := by
  have c67665 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67666 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67667 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67668 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67669 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67670 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_0
  simp only [← c67665, ← c67666, ← c67667] at e_1096_0
  have e_1096_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_1
  simp only [← c67668, ← c67669, k2, ← c67670] at e_1096_1
  have hr := e_1096_1
  simp only [e_1096_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9519)) := by
  have c67671 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67672 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67673 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67674 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67675 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67676 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_2
  simp only [← c67671, ← c67672, ← c67673] at e_1096_2
  have e_1096_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_3
  simp only [← c67674, ← c67675, k2, ← c67676] at e_1096_3
  have hr := e_1096_3
  simp only [e_1096_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9520)) := by
  have c67677 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67678 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67679 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67680 := (publicBatchWrapper86_copies2115 a h).1
  have c67681 := (publicBatchWrapper86_copies2115 a h).2.1
  have c67682 := (publicBatchWrapper86_copies2115 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_4
  simp only [← c67677, ← c67678, ← c67679] at e_1096_4
  have e_1096_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_5
  simp only [← c67680, ← c67681, k2, ← c67682] at e_1096_5
  have hr := e_1096_5
  simp only [e_1096_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9521)) := by
  have c67683 := (publicBatchWrapper86_copies2115 a h).2.2.2.1
  have c67684 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.1
  have c67685 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.1
  have c67686 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.1
  have c67687 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.1
  have c67688 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_6
  simp only [← c67683, ← c67684, ← c67685] at e_1096_6
  have e_1096_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_7
  simp only [← c67686, ← c67687, k2, ← c67688] at e_1096_7
  have hr := e_1096_7
  simp only [e_1096_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9522)) := by
  have c67689 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.1
  have c67690 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67691 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67692 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67693 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67694 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_8
  simp only [← c67689, ← c67690, ← c67691] at e_1096_8
  have e_1096_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_9
  simp only [← c67692, ← c67693, k2, ← c67694] at e_1096_9
  have hr := e_1096_9
  simp only [e_1096_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9523)) := by
  have c67695 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67696 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67697 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67698 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67699 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67700 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_10
  simp only [← c67695, ← c67696, ← c67697] at e_1096_10
  have e_1096_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_11
  simp only [← c67698, ← c67699, k2, ← c67700] at e_1096_11
  have hr := e_1096_11
  simp only [e_1096_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9524)) := by
  have c67701 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67702 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67703 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67704 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67705 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67706 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_12
  simp only [← c67701, ← c67702, ← c67703] at e_1096_12
  have e_1096_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_13
  simp only [← c67704, ← c67705, k2, ← c67706] at e_1096_13
  have hr := e_1096_13
  simp only [e_1096_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9525)) := by
  have c67707 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67708 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67709 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67710 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67711 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67712 := (publicBatchWrapper86_copies2116 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_14
  simp only [← c67707, ← c67708, ← c67709] at e_1096_14
  have e_1096_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_15
  simp only [← c67710, ← c67711, k2, ← c67712] at e_1096_15
  have hr := e_1096_15
  simp only [e_1096_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9526)) := by
  have c67713 := (publicBatchWrapper86_copies2116 a h).2.1
  have c67714 := (publicBatchWrapper86_copies2116 a h).2.2.1
  have c67715 := (publicBatchWrapper86_copies2116 a h).2.2.2.1
  have c67716 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.1
  have c67717 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.1
  have c67718 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_16
  simp only [← c67713, ← c67714, ← c67715] at e_1096_16
  have e_1096_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_17
  simp only [← c67716, ← c67717, k2, ← c67718] at e_1096_17
  have hr := e_1096_17
  simp only [e_1096_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9527)) := by
  have c67719 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.1
  have c67720 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.1
  have c67721 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.1
  have c67722 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67723 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67724 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_18
  simp only [← c67719, ← c67720, ← c67721] at e_1096_18
  have e_1096_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_19
  simp only [← c67722, ← c67723, k2, ← c67724] at e_1096_19
  have hr := e_1096_19
  simp only [e_1096_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9528)) := by
  have c67725 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67726 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67727 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67728 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67729 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67730 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_0
  simp only [← c67725, ← c67726, ← c67727] at e_1097_0
  have e_1097_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_1
  simp only [← c67728, ← c67729, k2, ← c67730] at e_1097_1
  have hr := e_1097_1
  simp only [e_1097_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9529)) := by
  have c67731 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67732 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67733 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67734 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67735 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67736 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_2
  simp only [← c67731, ← c67732, ← c67733] at e_1097_2
  have e_1097_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_3
  simp only [← c67734, ← c67735, k2, ← c67736] at e_1097_3
  have hr := e_1097_3
  simp only [e_1097_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9530)) := by
  have c67737 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67738 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67739 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67740 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67741 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67742 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_4
  simp only [← c67737, ← c67738, ← c67739] at e_1097_4
  have e_1097_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_5
  simp only [← c67740, ← c67741, k2, ← c67742] at e_1097_5
  have hr := e_1097_5
  simp only [e_1097_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9531)) := by
  have c67743 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67744 := (publicBatchWrapper86_copies2117 a h).1
  have c67745 := (publicBatchWrapper86_copies2117 a h).2.1
  have c67746 := (publicBatchWrapper86_copies2117 a h).2.2.1
  have c67747 := (publicBatchWrapper86_copies2117 a h).2.2.2.1
  have c67748 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_6
  simp only [← c67743, ← c67744, ← c67745] at e_1097_6
  have e_1097_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_7
  simp only [← c67746, ← c67747, k2, ← c67748] at e_1097_7
  have hr := e_1097_7
  simp only [e_1097_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9532)) := by
  have c67749 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.1
  have c67750 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.1
  have c67751 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.1
  have c67752 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.1
  have c67753 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.1
  have c67754 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_8
  simp only [← c67749, ← c67750, ← c67751] at e_1097_8
  have e_1097_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_9
  simp only [← c67752, ← c67753, k2, ← c67754] at e_1097_9
  have hr := e_1097_9
  simp only [e_1097_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9533)) := by
  have c67755 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67756 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67757 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67758 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67759 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67760 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_10
  simp only [← c67755, ← c67756, ← c67757] at e_1097_10
  have e_1097_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_11
  simp only [← c67758, ← c67759, k2, ← c67760] at e_1097_11
  have hr := e_1097_11
  simp only [e_1097_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9534)) := by
  have c67761 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67762 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67763 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67764 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67765 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67766 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_12
  simp only [← c67761, ← c67762, ← c67763] at e_1097_12
  have e_1097_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_13
  simp only [← c67764, ← c67765, k2, ← c67766] at e_1097_13
  have hr := e_1097_13
  simp only [e_1097_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9535)) := by
  have c67767 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67768 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67769 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67770 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67771 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67772 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_14
  simp only [← c67767, ← c67768, ← c67769] at e_1097_14
  have e_1097_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_15
  simp only [← c67770, ← c67771, k2, ← c67772] at e_1097_15
  have hr := e_1097_15
  simp only [e_1097_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
