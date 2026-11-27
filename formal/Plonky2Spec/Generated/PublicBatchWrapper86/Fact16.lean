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

theorem publicBatchWrapper86_f12108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11474)) := by
  have c77001 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.1
  have c77002 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77003 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77004 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77005 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77006 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_12
  simp only [← c77001, ← c77002, ← c77003] at e_1251_12
  have e_1251_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_13
  simp only [← c77004, ← c77005, k2, ← c77006] at e_1251_13
  have hr := e_1251_13
  simp only [e_1251_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11475)) := by
  have c77007 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77008 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77009 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77010 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77011 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77012 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_14
  simp only [← c77007, ← c77008, ← c77009] at e_1251_14
  have e_1251_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_15
  simp only [← c77010, ← c77011, k2, ← c77012] at e_1251_15
  have hr := e_1251_15
  simp only [e_1251_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11476)) := by
  have c77013 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77014 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77015 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77016 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77017 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77018 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_16
  simp only [← c77013, ← c77014, ← c77015] at e_1251_16
  have e_1251_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_17
  simp only [← c77016, ← c77017, k2, ← c77018] at e_1251_17
  have hr := e_1251_17
  simp only [e_1251_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11477)) := by
  have c77019 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77020 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77021 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77022 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77023 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77024 := (publicBatchWrapper86_copies2407 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_18
  simp only [← c77019, ← c77020, ← c77021] at e_1251_18
  have e_1251_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_19
  simp only [← c77022, ← c77023, k2, ← c77024] at e_1251_19
  have hr := e_1251_19
  simp only [e_1251_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11478)) := by
  have c77025 := (publicBatchWrapper86_copies2407 a h).2.1
  have c77026 := (publicBatchWrapper86_copies2407 a h).2.2.1
  have c77027 := (publicBatchWrapper86_copies2407 a h).2.2.2.1
  have c77028 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.1
  have c77029 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.1
  have c77030 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_0
  simp only [← c77025, ← c77026, ← c77027] at e_1252_0
  have e_1252_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_1
  simp only [← c77028, ← c77029, k2, ← c77030] at e_1252_1
  have hr := e_1252_1
  simp only [e_1252_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11479)) := by
  have c77031 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.1
  have c77032 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.1
  have c77033 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.1
  have c77034 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77035 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77036 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_2
  simp only [← c77031, ← c77032, ← c77033] at e_1252_2
  have e_1252_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_3
  simp only [← c77034, ← c77035, k2, ← c77036] at e_1252_3
  have hr := e_1252_3
  simp only [e_1252_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11480)) := by
  have c77037 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77038 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77039 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77040 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77041 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77042 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_4
  simp only [← c77037, ← c77038, ← c77039] at e_1252_4
  have e_1252_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_5
  simp only [← c77040, ← c77041, k2, ← c77042] at e_1252_5
  have hr := e_1252_5
  simp only [e_1252_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11481)) := by
  have c77043 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77044 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77045 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77046 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77047 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77048 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_6
  simp only [← c77043, ← c77044, ← c77045] at e_1252_6
  have e_1252_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_7
  simp only [← c77046, ← c77047, k2, ← c77048] at e_1252_7
  have hr := e_1252_7
  simp only [e_1252_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11482)) := by
  have c77049 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77050 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77051 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77052 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77053 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77054 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_8
  simp only [← c77049, ← c77050, ← c77051] at e_1252_8
  have e_1252_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_9
  simp only [← c77052, ← c77053, k2, ← c77054] at e_1252_9
  have hr := e_1252_9
  simp only [e_1252_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11483)) := by
  have c77055 := (publicBatchWrapper86_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77056 := (publicBatchWrapper86_copies2408 a h).1
  have c77057 := (publicBatchWrapper86_copies2408 a h).2.1
  have c77058 := (publicBatchWrapper86_copies2408 a h).2.2.1
  have c77059 := (publicBatchWrapper86_copies2408 a h).2.2.2.1
  have c77060 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_10
  simp only [← c77055, ← c77056, ← c77057] at e_1252_10
  have e_1252_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_11
  simp only [← c77058, ← c77059, k2, ← c77060] at e_1252_11
  have hr := e_1252_11
  simp only [e_1252_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11484)) := by
  have c77061 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.1
  have c77062 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.1
  have c77063 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.1
  have c77064 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.1
  have c77065 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.1
  have c77066 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_12
  simp only [← c77061, ← c77062, ← c77063] at e_1252_12
  have e_1252_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_13
  simp only [← c77064, ← c77065, k2, ← c77066] at e_1252_13
  have hr := e_1252_13
  simp only [e_1252_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11485)) := by
  have c77067 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77068 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77069 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77070 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77071 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77072 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_14
  simp only [← c77067, ← c77068, ← c77069] at e_1252_14
  have e_1252_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_15
  simp only [← c77070, ← c77071, k2, ← c77072] at e_1252_15
  have hr := e_1252_15
  simp only [e_1252_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11486)) := by
  have c77073 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77074 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77075 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77076 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77077 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77078 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_16
  simp only [← c77073, ← c77074, ← c77075] at e_1252_16
  have e_1252_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_17
  simp only [← c77076, ← c77077, k2, ← c77078] at e_1252_17
  have hr := e_1252_17
  simp only [e_1252_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1252 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11487)) := by
  have c77079 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77080 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77081 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77082 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77083 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77084 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1252_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_18
  simp only [← c77079, ← c77080, ← c77081] at e_1252_18
  have e_1252_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_19
  simp only [← c77082, ← c77083, k2, ← c77084] at e_1252_19
  have hr := e_1252_19
  simp only [e_1252_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11488)) := by
  have c77085 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77086 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77087 := (publicBatchWrapper86_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77088 := (publicBatchWrapper86_copies2409 a h).1
  have c77089 := (publicBatchWrapper86_copies2409 a h).2.1
  have c77090 := (publicBatchWrapper86_copies2409 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_0
  simp only [← c77085, ← c77086, ← c77087] at e_1253_0
  have e_1253_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_1
  simp only [← c77088, ← c77089, k2, ← c77090] at e_1253_1
  have hr := e_1253_1
  simp only [e_1253_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11489)) := by
  have c77091 := (publicBatchWrapper86_copies2409 a h).2.2.2.1
  have c77092 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.1
  have c77093 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.1
  have c77094 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.1
  have c77095 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.1
  have c77096 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_2
  simp only [← c77091, ← c77092, ← c77093] at e_1253_2
  have e_1253_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_3
  simp only [← c77094, ← c77095, k2, ← c77096] at e_1253_3
  have hr := e_1253_3
  simp only [e_1253_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11490)) := by
  have c77097 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.1
  have c77098 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77099 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77100 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77101 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77102 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_4
  simp only [← c77097, ← c77098, ← c77099] at e_1253_4
  have e_1253_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_5
  simp only [← c77100, ← c77101, k2, ← c77102] at e_1253_5
  have hr := e_1253_5
  simp only [e_1253_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11491)) := by
  have c77103 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77104 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77105 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77106 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77107 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77108 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_6
  simp only [← c77103, ← c77104, ← c77105] at e_1253_6
  have e_1253_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_7
  simp only [← c77106, ← c77107, k2, ← c77108] at e_1253_7
  have hr := e_1253_7
  simp only [e_1253_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11492)) := by
  have c77109 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77110 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77111 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77112 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77113 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77114 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_8
  simp only [← c77109, ← c77110, ← c77111] at e_1253_8
  have e_1253_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_9
  simp only [← c77112, ← c77113, k2, ← c77114] at e_1253_9
  have hr := e_1253_9
  simp only [e_1253_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11493)) := by
  have c77115 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77116 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77117 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77118 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77119 := (publicBatchWrapper86_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77120 := (publicBatchWrapper86_copies2410 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_10
  simp only [← c77115, ← c77116, ← c77117] at e_1253_10
  have e_1253_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_11
  simp only [← c77118, ← c77119, k2, ← c77120] at e_1253_11
  have hr := e_1253_11
  simp only [e_1253_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11494)) := by
  have c77121 := (publicBatchWrapper86_copies2410 a h).2.1
  have c77122 := (publicBatchWrapper86_copies2410 a h).2.2.1
  have c77123 := (publicBatchWrapper86_copies2410 a h).2.2.2.1
  have c77124 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.1
  have c77125 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.1
  have c77126 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_12
  simp only [← c77121, ← c77122, ← c77123] at e_1253_12
  have e_1253_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_13
  simp only [← c77124, ← c77125, k2, ← c77126] at e_1253_13
  have hr := e_1253_13
  simp only [e_1253_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11495)) := by
  have c77127 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.1
  have c77128 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.1
  have c77129 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.1
  have c77130 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77131 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77132 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_14
  simp only [← c77127, ← c77128, ← c77129] at e_1253_14
  have e_1253_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_15
  simp only [← c77130, ← c77131, k2, ← c77132] at e_1253_15
  have hr := e_1253_15
  simp only [e_1253_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11496)) := by
  have c77133 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77134 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77135 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77136 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77137 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77138 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_16
  simp only [← c77133, ← c77134, ← c77135] at e_1253_16
  have e_1253_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_17
  simp only [← c77136, ← c77137, k2, ← c77138] at e_1253_17
  have hr := e_1253_17
  simp only [e_1253_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1253 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11497)) := by
  have c77139 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77140 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77141 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77142 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77143 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77144 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1253_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_18
  simp only [← c77139, ← c77140, ← c77141] at e_1253_18
  have e_1253_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_19
  simp only [← c77142, ← c77143, k2, ← c77144] at e_1253_19
  have hr := e_1253_19
  simp only [e_1253_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11498)) := by
  have c77145 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77146 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77147 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77148 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77149 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77150 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_0
  simp only [← c77145, ← c77146, ← c77147] at e_1254_0
  have e_1254_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_1
  simp only [← c77148, ← c77149, k2, ← c77150] at e_1254_1
  have hr := e_1254_1
  simp only [e_1254_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11499)) := by
  have c77151 := (publicBatchWrapper86_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77152 := (publicBatchWrapper86_copies2411 a h).1
  have c77153 := (publicBatchWrapper86_copies2411 a h).2.1
  have c77154 := (publicBatchWrapper86_copies2411 a h).2.2.1
  have c77155 := (publicBatchWrapper86_copies2411 a h).2.2.2.1
  have c77156 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_2
  simp only [← c77151, ← c77152, ← c77153] at e_1254_2
  have e_1254_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_3
  simp only [← c77154, ← c77155, k2, ← c77156] at e_1254_3
  have hr := e_1254_3
  simp only [e_1254_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11500)) := by
  have c77157 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.1
  have c77158 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.1
  have c77159 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.1
  have c77160 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.1
  have c77161 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.1
  have c77162 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_4
  simp only [← c77157, ← c77158, ← c77159] at e_1254_4
  have e_1254_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_5
  simp only [← c77160, ← c77161, k2, ← c77162] at e_1254_5
  have hr := e_1254_5
  simp only [e_1254_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11501)) := by
  have c77163 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77164 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77165 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77166 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77167 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77168 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_6
  simp only [← c77163, ← c77164, ← c77165] at e_1254_6
  have e_1254_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_7
  simp only [← c77166, ← c77167, k2, ← c77168] at e_1254_7
  have hr := e_1254_7
  simp only [e_1254_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11502)) := by
  have c77169 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77170 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77171 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77172 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77173 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77174 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_8
  simp only [← c77169, ← c77170, ← c77171] at e_1254_8
  have e_1254_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_9
  simp only [← c77172, ← c77173, k2, ← c77174] at e_1254_9
  have hr := e_1254_9
  simp only [e_1254_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11503)) := by
  have c77175 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77176 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77177 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77178 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77179 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77180 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_10
  simp only [← c77175, ← c77176, ← c77177] at e_1254_10
  have e_1254_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_11
  simp only [← c77178, ← c77179, k2, ← c77180] at e_1254_11
  have hr := e_1254_11
  simp only [e_1254_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11504)) := by
  have c77181 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77182 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77183 := (publicBatchWrapper86_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77184 := (publicBatchWrapper86_copies2412 a h).1
  have c77185 := (publicBatchWrapper86_copies2412 a h).2.1
  have c77186 := (publicBatchWrapper86_copies2412 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_12
  simp only [← c77181, ← c77182, ← c77183] at e_1254_12
  have e_1254_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_13
  simp only [← c77184, ← c77185, k2, ← c77186] at e_1254_13
  have hr := e_1254_13
  simp only [e_1254_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11505)) := by
  have c77187 := (publicBatchWrapper86_copies2412 a h).2.2.2.1
  have c77188 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.1
  have c77189 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.1
  have c77190 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.1
  have c77191 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.1
  have c77192 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_14
  simp only [← c77187, ← c77188, ← c77189] at e_1254_14
  have e_1254_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_15
  simp only [← c77190, ← c77191, k2, ← c77192] at e_1254_15
  have hr := e_1254_15
  simp only [e_1254_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11506)) := by
  have c77193 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.1
  have c77194 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77195 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77196 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77197 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77198 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_16
  simp only [← c77193, ← c77194, ← c77195] at e_1254_16
  have e_1254_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_17
  simp only [← c77196, ← c77197, k2, ← c77198] at e_1254_17
  have hr := e_1254_17
  simp only [e_1254_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1254 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11507)) := by
  have c77199 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77200 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77201 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77202 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77203 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77204 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1254_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_18
  simp only [← c77199, ← c77200, ← c77201] at e_1254_18
  have e_1254_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_19
  simp only [← c77202, ← c77203, k2, ← c77204] at e_1254_19
  have hr := e_1254_19
  simp only [e_1254_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11508)) := by
  have c77205 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77206 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77207 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77208 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77209 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77210 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_0
  simp only [← c77205, ← c77206, ← c77207] at e_1255_0
  have e_1255_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_1
  simp only [← c77208, ← c77209, k2, ← c77210] at e_1255_1
  have hr := e_1255_1
  simp only [e_1255_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11509)) := by
  have c77211 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77212 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77213 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77214 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77215 := (publicBatchWrapper86_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77216 := (publicBatchWrapper86_copies2413 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_2
  simp only [← c77211, ← c77212, ← c77213] at e_1255_2
  have e_1255_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_3
  simp only [← c77214, ← c77215, k2, ← c77216] at e_1255_3
  have hr := e_1255_3
  simp only [e_1255_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11510)) := by
  have c77217 := (publicBatchWrapper86_copies2413 a h).2.1
  have c77218 := (publicBatchWrapper86_copies2413 a h).2.2.1
  have c77219 := (publicBatchWrapper86_copies2413 a h).2.2.2.1
  have c77220 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.1
  have c77221 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.1
  have c77222 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_4
  simp only [← c77217, ← c77218, ← c77219] at e_1255_4
  have e_1255_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_5
  simp only [← c77220, ← c77221, k2, ← c77222] at e_1255_5
  have hr := e_1255_5
  simp only [e_1255_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11511)) := by
  have c77223 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.1
  have c77224 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.1
  have c77225 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.1
  have c77226 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77227 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77228 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_6
  simp only [← c77223, ← c77224, ← c77225] at e_1255_6
  have e_1255_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_7
  simp only [← c77226, ← c77227, k2, ← c77228] at e_1255_7
  have hr := e_1255_7
  simp only [e_1255_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11512)) := by
  have c77229 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77230 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77231 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77232 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77233 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77234 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_8
  simp only [← c77229, ← c77230, ← c77231] at e_1255_8
  have e_1255_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_9
  simp only [← c77232, ← c77233, k2, ← c77234] at e_1255_9
  have hr := e_1255_9
  simp only [e_1255_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11513)) := by
  have c77235 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77236 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77237 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77238 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77239 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77240 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_10
  simp only [← c77235, ← c77236, ← c77237] at e_1255_10
  have e_1255_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_11
  simp only [← c77238, ← c77239, k2, ← c77240] at e_1255_11
  have hr := e_1255_11
  simp only [e_1255_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11514)) := by
  have c77241 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77242 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77243 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77244 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77245 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77246 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_12
  simp only [← c77241, ← c77242, ← c77243] at e_1255_12
  have e_1255_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_13
  simp only [← c77244, ← c77245, k2, ← c77246] at e_1255_13
  have hr := e_1255_13
  simp only [e_1255_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11515)) := by
  have c77247 := (publicBatchWrapper86_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77248 := (publicBatchWrapper86_copies2414 a h).1
  have c77249 := (publicBatchWrapper86_copies2414 a h).2.1
  have c77250 := (publicBatchWrapper86_copies2414 a h).2.2.1
  have c77251 := (publicBatchWrapper86_copies2414 a h).2.2.2.1
  have c77252 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_14
  simp only [← c77247, ← c77248, ← c77249] at e_1255_14
  have e_1255_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_15
  simp only [← c77250, ← c77251, k2, ← c77252] at e_1255_15
  have hr := e_1255_15
  simp only [e_1255_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11516)) := by
  have c77253 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.1
  have c77254 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.1
  have c77255 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.1
  have c77256 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.1
  have c77257 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.1
  have c77258 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_16
  simp only [← c77253, ← c77254, ← c77255] at e_1255_16
  have e_1255_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_17
  simp only [← c77256, ← c77257, k2, ← c77258] at e_1255_17
  have hr := e_1255_17
  simp only [e_1255_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1255 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11517)) := by
  have c77259 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77260 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77261 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77262 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77263 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77264 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1255_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_18
  simp only [← c77259, ← c77260, ← c77261] at e_1255_18
  have e_1255_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_19
  simp only [← c77262, ← c77263, k2, ← c77264] at e_1255_19
  have hr := e_1255_19
  simp only [e_1255_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11518)) := by
  have c77265 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77266 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77267 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77268 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77269 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77270 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_0
  simp only [← c77265, ← c77266, ← c77267] at e_1256_0
  have e_1256_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_1
  simp only [← c77268, ← c77269, k2, ← c77270] at e_1256_1
  have hr := e_1256_1
  simp only [e_1256_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11519)) := by
  have c77271 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77272 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77273 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77274 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77275 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77276 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_2
  simp only [← c77271, ← c77272, ← c77273] at e_1256_2
  have e_1256_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_3
  simp only [← c77274, ← c77275, k2, ← c77276] at e_1256_3
  have hr := e_1256_3
  simp only [e_1256_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11520)) := by
  have c77277 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77278 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77279 := (publicBatchWrapper86_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77280 := (publicBatchWrapper86_copies2415 a h).1
  have c77281 := (publicBatchWrapper86_copies2415 a h).2.1
  have c77282 := (publicBatchWrapper86_copies2415 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_4
  simp only [← c77277, ← c77278, ← c77279] at e_1256_4
  have e_1256_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_5
  simp only [← c77280, ← c77281, k2, ← c77282] at e_1256_5
  have hr := e_1256_5
  simp only [e_1256_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11521)) := by
  have c77283 := (publicBatchWrapper86_copies2415 a h).2.2.2.1
  have c77284 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.1
  have c77285 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.1
  have c77286 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.1
  have c77287 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.1
  have c77288 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_6
  simp only [← c77283, ← c77284, ← c77285] at e_1256_6
  have e_1256_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_7
  simp only [← c77286, ← c77287, k2, ← c77288] at e_1256_7
  have hr := e_1256_7
  simp only [e_1256_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11522)) := by
  have c77289 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.1
  have c77290 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77291 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77292 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77293 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77294 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_8
  simp only [← c77289, ← c77290, ← c77291] at e_1256_8
  have e_1256_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_9
  simp only [← c77292, ← c77293, k2, ← c77294] at e_1256_9
  have hr := e_1256_9
  simp only [e_1256_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11523)) := by
  have c77295 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77296 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77297 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77298 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77299 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77300 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_10
  simp only [← c77295, ← c77296, ← c77297] at e_1256_10
  have e_1256_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_11
  simp only [← c77298, ← c77299, k2, ← c77300] at e_1256_11
  have hr := e_1256_11
  simp only [e_1256_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11524)) := by
  have c77301 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77302 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77303 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77304 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77305 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77306 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_12
  simp only [← c77301, ← c77302, ← c77303] at e_1256_12
  have e_1256_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_13
  simp only [← c77304, ← c77305, k2, ← c77306] at e_1256_13
  have hr := e_1256_13
  simp only [e_1256_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11525)) := by
  have c77307 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77308 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77309 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77310 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77311 := (publicBatchWrapper86_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77312 := (publicBatchWrapper86_copies2416 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_14
  simp only [← c77307, ← c77308, ← c77309] at e_1256_14
  have e_1256_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_15
  simp only [← c77310, ← c77311, k2, ← c77312] at e_1256_15
  have hr := e_1256_15
  simp only [e_1256_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11526)) := by
  have c77313 := (publicBatchWrapper86_copies2416 a h).2.1
  have c77314 := (publicBatchWrapper86_copies2416 a h).2.2.1
  have c77315 := (publicBatchWrapper86_copies2416 a h).2.2.2.1
  have c77316 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.1
  have c77317 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.1
  have c77318 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_16
  simp only [← c77313, ← c77314, ← c77315] at e_1256_16
  have e_1256_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_17
  simp only [← c77316, ← c77317, k2, ← c77318] at e_1256_17
  have hr := e_1256_17
  simp only [e_1256_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1256 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11527)) := by
  have c77319 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.1
  have c77320 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.1
  have c77321 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.1
  have c77322 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77323 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77324 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1256_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_18
  simp only [← c77319, ← c77320, ← c77321] at e_1256_18
  have e_1256_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_19
  simp only [← c77322, ← c77323, k2, ← c77324] at e_1256_19
  have hr := e_1256_19
  simp only [e_1256_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11528)) := by
  have c77325 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77326 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77327 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77328 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77329 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77330 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_0
  simp only [← c77325, ← c77326, ← c77327] at e_1257_0
  have e_1257_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_1
  simp only [← c77328, ← c77329, k2, ← c77330] at e_1257_1
  have hr := e_1257_1
  simp only [e_1257_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11529)) := by
  have c77331 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77332 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77333 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77334 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77335 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77336 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_2
  simp only [← c77331, ← c77332, ← c77333] at e_1257_2
  have e_1257_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_3
  simp only [← c77334, ← c77335, k2, ← c77336] at e_1257_3
  have hr := e_1257_3
  simp only [e_1257_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11530)) := by
  have c77337 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77338 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77339 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77340 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77341 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77342 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_4
  simp only [← c77337, ← c77338, ← c77339] at e_1257_4
  have e_1257_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_5
  simp only [← c77340, ← c77341, k2, ← c77342] at e_1257_5
  have hr := e_1257_5
  simp only [e_1257_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11531)) := by
  have c77343 := (publicBatchWrapper86_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77344 := (publicBatchWrapper86_copies2417 a h).1
  have c77345 := (publicBatchWrapper86_copies2417 a h).2.1
  have c77346 := (publicBatchWrapper86_copies2417 a h).2.2.1
  have c77347 := (publicBatchWrapper86_copies2417 a h).2.2.2.1
  have c77348 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_6
  simp only [← c77343, ← c77344, ← c77345] at e_1257_6
  have e_1257_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_7
  simp only [← c77346, ← c77347, k2, ← c77348] at e_1257_7
  have hr := e_1257_7
  simp only [e_1257_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11532)) := by
  have c77349 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.1
  have c77350 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.1
  have c77351 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.1
  have c77352 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.1
  have c77353 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.1
  have c77354 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_8
  simp only [← c77349, ← c77350, ← c77351] at e_1257_8
  have e_1257_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_9
  simp only [← c77352, ← c77353, k2, ← c77354] at e_1257_9
  have hr := e_1257_9
  simp only [e_1257_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11533)) := by
  have c77355 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77356 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77357 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77358 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77359 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77360 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_10
  simp only [← c77355, ← c77356, ← c77357] at e_1257_10
  have e_1257_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_11
  simp only [← c77358, ← c77359, k2, ← c77360] at e_1257_11
  have hr := e_1257_11
  simp only [e_1257_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11534)) := by
  have c77361 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77362 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77363 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77364 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77365 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77366 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_12
  simp only [← c77361, ← c77362, ← c77363] at e_1257_12
  have e_1257_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_13
  simp only [← c77364, ← c77365, k2, ← c77366] at e_1257_13
  have hr := e_1257_13
  simp only [e_1257_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11535)) := by
  have c77367 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77368 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77369 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77370 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77371 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77372 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_14
  simp only [← c77367, ← c77368, ← c77369] at e_1257_14
  have e_1257_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_15
  simp only [← c77370, ← c77371, k2, ← c77372] at e_1257_15
  have hr := e_1257_15
  simp only [e_1257_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11536)) := by
  have c77373 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77374 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77375 := (publicBatchWrapper86_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77376 := (publicBatchWrapper86_copies2418 a h).1
  have c77377 := (publicBatchWrapper86_copies2418 a h).2.1
  have c77378 := (publicBatchWrapper86_copies2418 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_16
  simp only [← c77373, ← c77374, ← c77375] at e_1257_16
  have e_1257_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_17
  simp only [← c77376, ← c77377, k2, ← c77378] at e_1257_17
  have hr := e_1257_17
  simp only [e_1257_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1257 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11537)) := by
  have c77379 := (publicBatchWrapper86_copies2418 a h).2.2.2.1
  have c77380 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.1
  have c77381 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.1
  have c77382 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.1
  have c77383 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.1
  have c77384 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1257_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_18
  simp only [← c77379, ← c77380, ← c77381] at e_1257_18
  have e_1257_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_19
  simp only [← c77382, ← c77383, k2, ← c77384] at e_1257_19
  have hr := e_1257_19
  simp only [e_1257_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11538)) := by
  have c77385 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.1
  have c77386 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77387 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77388 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77389 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77390 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_0
  simp only [← c77385, ← c77386, ← c77387] at e_1258_0
  have e_1258_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_1
  simp only [← c77388, ← c77389, k2, ← c77390] at e_1258_1
  have hr := e_1258_1
  simp only [e_1258_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11539)) := by
  have c77391 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77392 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77393 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77394 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77395 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77396 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_2
  simp only [← c77391, ← c77392, ← c77393] at e_1258_2
  have e_1258_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_3
  simp only [← c77394, ← c77395, k2, ← c77396] at e_1258_3
  have hr := e_1258_3
  simp only [e_1258_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11540)) := by
  have c77397 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77398 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77399 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77400 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77401 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77402 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_4
  simp only [← c77397, ← c77398, ← c77399] at e_1258_4
  have e_1258_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_5
  simp only [← c77400, ← c77401, k2, ← c77402] at e_1258_5
  have hr := e_1258_5
  simp only [e_1258_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11541)) := by
  have c77403 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77404 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77405 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77406 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77407 := (publicBatchWrapper86_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77408 := (publicBatchWrapper86_copies2419 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_6
  simp only [← c77403, ← c77404, ← c77405] at e_1258_6
  have e_1258_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_7
  simp only [← c77406, ← c77407, k2, ← c77408] at e_1258_7
  have hr := e_1258_7
  simp only [e_1258_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11542)) := by
  have c77409 := (publicBatchWrapper86_copies2419 a h).2.1
  have c77410 := (publicBatchWrapper86_copies2419 a h).2.2.1
  have c77411 := (publicBatchWrapper86_copies2419 a h).2.2.2.1
  have c77412 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.1
  have c77413 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.1
  have c77414 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_8
  simp only [← c77409, ← c77410, ← c77411] at e_1258_8
  have e_1258_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_9
  simp only [← c77412, ← c77413, k2, ← c77414] at e_1258_9
  have hr := e_1258_9
  simp only [e_1258_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11543)) := by
  have c77415 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.1
  have c77416 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.1
  have c77417 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.1
  have c77418 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77419 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77420 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_10
  simp only [← c77415, ← c77416, ← c77417] at e_1258_10
  have e_1258_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_11
  simp only [← c77418, ← c77419, k2, ← c77420] at e_1258_11
  have hr := e_1258_11
  simp only [e_1258_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11544)) := by
  have c77421 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77422 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77423 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77424 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77425 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77426 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_12
  simp only [← c77421, ← c77422, ← c77423] at e_1258_12
  have e_1258_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_13
  simp only [← c77424, ← c77425, k2, ← c77426] at e_1258_13
  have hr := e_1258_13
  simp only [e_1258_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11545)) := by
  have c77427 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77428 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77429 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77430 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77431 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77432 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_14
  simp only [← c77427, ← c77428, ← c77429] at e_1258_14
  have e_1258_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_15
  simp only [← c77430, ← c77431, k2, ← c77432] at e_1258_15
  have hr := e_1258_15
  simp only [e_1258_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11546)) := by
  have c77433 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77434 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77435 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77436 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77437 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77438 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_16
  simp only [← c77433, ← c77434, ← c77435] at e_1258_16
  have e_1258_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_17
  simp only [← c77436, ← c77437, k2, ← c77438] at e_1258_17
  have hr := e_1258_17
  simp only [e_1258_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1258 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11547)) := by
  have c77439 := (publicBatchWrapper86_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77440 := (publicBatchWrapper86_copies2420 a h).1
  have c77441 := (publicBatchWrapper86_copies2420 a h).2.1
  have c77442 := (publicBatchWrapper86_copies2420 a h).2.2.1
  have c77443 := (publicBatchWrapper86_copies2420 a h).2.2.2.1
  have c77444 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1258_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_18
  simp only [← c77439, ← c77440, ← c77441] at e_1258_18
  have e_1258_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_19
  simp only [← c77442, ← c77443, k2, ← c77444] at e_1258_19
  have hr := e_1258_19
  simp only [e_1258_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11548)) := by
  have c77445 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.1
  have c77446 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.1
  have c77447 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.1
  have c77448 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.1
  have c77449 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.1
  have c77450 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_0
  simp only [← c77445, ← c77446, ← c77447] at e_1259_0
  have e_1259_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_1
  simp only [← c77448, ← c77449, k2, ← c77450] at e_1259_1
  have hr := e_1259_1
  simp only [e_1259_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11549)) := by
  have c77451 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77452 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77453 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77454 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77455 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77456 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_2
  simp only [← c77451, ← c77452, ← c77453] at e_1259_2
  have e_1259_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_3
  simp only [← c77454, ← c77455, k2, ← c77456] at e_1259_3
  have hr := e_1259_3
  simp only [e_1259_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11550)) := by
  have c77457 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77458 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77459 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77460 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77461 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77462 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_4
  simp only [← c77457, ← c77458, ← c77459] at e_1259_4
  have e_1259_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_5
  simp only [← c77460, ← c77461, k2, ← c77462] at e_1259_5
  have hr := e_1259_5
  simp only [e_1259_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11551)) := by
  have c77463 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77464 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77465 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77466 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77467 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77468 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_6
  simp only [← c77463, ← c77464, ← c77465] at e_1259_6
  have e_1259_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_7
  simp only [← c77466, ← c77467, k2, ← c77468] at e_1259_7
  have hr := e_1259_7
  simp only [e_1259_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11552)) := by
  have c77469 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77470 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77471 := (publicBatchWrapper86_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77472 := (publicBatchWrapper86_copies2421 a h).1
  have c77473 := (publicBatchWrapper86_copies2421 a h).2.1
  have c77474 := (publicBatchWrapper86_copies2421 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_8
  simp only [← c77469, ← c77470, ← c77471] at e_1259_8
  have e_1259_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_9
  simp only [← c77472, ← c77473, k2, ← c77474] at e_1259_9
  have hr := e_1259_9
  simp only [e_1259_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11553)) := by
  have c77475 := (publicBatchWrapper86_copies2421 a h).2.2.2.1
  have c77476 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.1
  have c77477 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.1
  have c77478 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.1
  have c77479 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.1
  have c77480 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_10
  simp only [← c77475, ← c77476, ← c77477] at e_1259_10
  have e_1259_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_11
  simp only [← c77478, ← c77479, k2, ← c77480] at e_1259_11
  have hr := e_1259_11
  simp only [e_1259_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11554)) := by
  have c77481 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.1
  have c77482 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77483 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77484 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77485 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77486 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_12
  simp only [← c77481, ← c77482, ← c77483] at e_1259_12
  have e_1259_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_13
  simp only [← c77484, ← c77485, k2, ← c77486] at e_1259_13
  have hr := e_1259_13
  simp only [e_1259_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11555)) := by
  have c77487 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77488 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77489 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77490 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77491 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77492 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_14
  simp only [← c77487, ← c77488, ← c77489] at e_1259_14
  have e_1259_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_15
  simp only [← c77490, ← c77491, k2, ← c77492] at e_1259_15
  have hr := e_1259_15
  simp only [e_1259_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11556)) := by
  have c77493 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77494 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77495 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77496 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77497 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77498 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_16
  simp only [← c77493, ← c77494, ← c77495] at e_1259_16
  have e_1259_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_17
  simp only [← c77496, ← c77497, k2, ← c77498] at e_1259_17
  have hr := e_1259_17
  simp only [e_1259_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1259 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11557)) := by
  have c77499 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77500 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77501 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77502 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77503 := (publicBatchWrapper86_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77504 := (publicBatchWrapper86_copies2422 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1259_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_18
  simp only [← c77499, ← c77500, ← c77501] at e_1259_18
  have e_1259_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_19
  simp only [← c77502, ← c77503, k2, ← c77504] at e_1259_19
  have hr := e_1259_19
  simp only [e_1259_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11558)) := by
  have c77505 := (publicBatchWrapper86_copies2422 a h).2.1
  have c77506 := (publicBatchWrapper86_copies2422 a h).2.2.1
  have c77507 := (publicBatchWrapper86_copies2422 a h).2.2.2.1
  have c77508 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.1
  have c77509 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.1
  have c77510 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_0
  simp only [← c77505, ← c77506, ← c77507] at e_1260_0
  have e_1260_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_1
  simp only [← c77508, ← c77509, k2, ← c77510] at e_1260_1
  have hr := e_1260_1
  simp only [e_1260_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11559)) := by
  have c77511 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.1
  have c77512 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.1
  have c77513 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.1
  have c77514 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77515 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77516 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_2
  simp only [← c77511, ← c77512, ← c77513] at e_1260_2
  have e_1260_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_3
  simp only [← c77514, ← c77515, k2, ← c77516] at e_1260_3
  have hr := e_1260_3
  simp only [e_1260_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11560)) := by
  have c77517 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77518 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77519 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77520 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77521 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77522 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_4
  simp only [← c77517, ← c77518, ← c77519] at e_1260_4
  have e_1260_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_5
  simp only [← c77520, ← c77521, k2, ← c77522] at e_1260_5
  have hr := e_1260_5
  simp only [e_1260_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11561)) := by
  have c77523 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77524 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77525 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77526 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77527 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77528 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_6
  simp only [← c77523, ← c77524, ← c77525] at e_1260_6
  have e_1260_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_7
  simp only [← c77526, ← c77527, k2, ← c77528] at e_1260_7
  have hr := e_1260_7
  simp only [e_1260_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11562)) := by
  have c77529 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77530 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77531 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77532 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77533 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77534 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_8
  simp only [← c77529, ← c77530, ← c77531] at e_1260_8
  have e_1260_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_9
  simp only [← c77532, ← c77533, k2, ← c77534] at e_1260_9
  have hr := e_1260_9
  simp only [e_1260_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11563)) := by
  have c77535 := (publicBatchWrapper86_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77536 := (publicBatchWrapper86_copies2423 a h).1
  have c77537 := (publicBatchWrapper86_copies2423 a h).2.1
  have c77538 := (publicBatchWrapper86_copies2423 a h).2.2.1
  have c77539 := (publicBatchWrapper86_copies2423 a h).2.2.2.1
  have c77540 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_10
  simp only [← c77535, ← c77536, ← c77537] at e_1260_10
  have e_1260_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_11
  simp only [← c77538, ← c77539, k2, ← c77540] at e_1260_11
  have hr := e_1260_11
  simp only [e_1260_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11564)) := by
  have c77541 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.1
  have c77542 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.1
  have c77543 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.1
  have c77544 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.1
  have c77545 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.1
  have c77546 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_12
  simp only [← c77541, ← c77542, ← c77543] at e_1260_12
  have e_1260_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_13
  simp only [← c77544, ← c77545, k2, ← c77546] at e_1260_13
  have hr := e_1260_13
  simp only [e_1260_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11565)) := by
  have c77547 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77548 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77549 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77550 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77551 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77552 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_14
  simp only [← c77547, ← c77548, ← c77549] at e_1260_14
  have e_1260_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_15
  simp only [← c77550, ← c77551, k2, ← c77552] at e_1260_15
  have hr := e_1260_15
  simp only [e_1260_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11566)) := by
  have c77553 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77554 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77555 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77556 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77557 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77558 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_16
  simp only [← c77553, ← c77554, ← c77555] at e_1260_16
  have e_1260_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_17
  simp only [← c77556, ← c77557, k2, ← c77558] at e_1260_17
  have hr := e_1260_17
  simp only [e_1260_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1260 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11567)) := by
  have c77559 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77560 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77561 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77562 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77563 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77564 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1260_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_18
  simp only [← c77559, ← c77560, ← c77561] at e_1260_18
  have e_1260_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_19
  simp only [← c77562, ← c77563, k2, ← c77564] at e_1260_19
  have hr := e_1260_19
  simp only [e_1260_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11568)) := by
  have c77565 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77566 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77567 := (publicBatchWrapper86_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77568 := (publicBatchWrapper86_copies2424 a h).1
  have c77569 := (publicBatchWrapper86_copies2424 a h).2.1
  have c77570 := (publicBatchWrapper86_copies2424 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_0
  simp only [← c77565, ← c77566, ← c77567] at e_1261_0
  have e_1261_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_1
  simp only [← c77568, ← c77569, k2, ← c77570] at e_1261_1
  have hr := e_1261_1
  simp only [e_1261_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11569)) := by
  have c77571 := (publicBatchWrapper86_copies2424 a h).2.2.2.1
  have c77572 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.1
  have c77573 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.1
  have c77574 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.1
  have c77575 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.1
  have c77576 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_2
  simp only [← c77571, ← c77572, ← c77573] at e_1261_2
  have e_1261_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_3
  simp only [← c77574, ← c77575, k2, ← c77576] at e_1261_3
  have hr := e_1261_3
  simp only [e_1261_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11610)) := by
  have c77577 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.1
  have c77578 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77579 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77580 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77581 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77582 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_4
  simp only [← c77577, ← c77578, ← c77579] at e_1261_4
  have e_1261_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_5
  simp only [← c77580, ← c77581, k2, ← c77582] at e_1261_5
  have hr := e_1261_5
  simp only [e_1261_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11611)) := by
  have c77583 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77584 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77585 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77586 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77587 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77588 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_6
  simp only [← c77583, ← c77584, ← c77585] at e_1261_6
  have e_1261_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_7
  simp only [← c77586, ← c77587, k2, ← c77588] at e_1261_7
  have hr := e_1261_7
  simp only [e_1261_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11612)) := by
  have c77589 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77590 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77591 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77592 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77593 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77594 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_8
  simp only [← c77589, ← c77590, ← c77591] at e_1261_8
  have e_1261_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_9
  simp only [← c77592, ← c77593, k2, ← c77594] at e_1261_9
  have hr := e_1261_9
  simp only [e_1261_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11613)) := by
  have c77595 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77596 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77597 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77598 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77599 := (publicBatchWrapper86_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77600 := (publicBatchWrapper86_copies2425 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_10
  simp only [← c77595, ← c77596, ← c77597] at e_1261_10
  have e_1261_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_11
  simp only [← c77598, ← c77599, k2, ← c77600] at e_1261_11
  have hr := e_1261_11
  simp only [e_1261_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11614)) := by
  have c77601 := (publicBatchWrapper86_copies2425 a h).2.1
  have c77602 := (publicBatchWrapper86_copies2425 a h).2.2.1
  have c77603 := (publicBatchWrapper86_copies2425 a h).2.2.2.1
  have c77604 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.1
  have c77605 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.1
  have c77606 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_12
  simp only [← c77601, ← c77602, ← c77603] at e_1261_12
  have e_1261_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_13
  simp only [← c77604, ← c77605, k2, ← c77606] at e_1261_13
  have hr := e_1261_13
  simp only [e_1261_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11615)) := by
  have c77607 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.1
  have c77608 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.1
  have c77609 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.1
  have c77610 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77611 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77612 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_14
  simp only [← c77607, ← c77608, ← c77609] at e_1261_14
  have e_1261_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_15
  simp only [← c77610, ← c77611, k2, ← c77612] at e_1261_15
  have hr := e_1261_15
  simp only [e_1261_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11616)) := by
  have c77613 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77614 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77615 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77616 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77617 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77618 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_16
  simp only [← c77613, ← c77614, ← c77615] at e_1261_16
  have e_1261_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_17
  simp only [← c77616, ← c77617, k2, ← c77618] at e_1261_17
  have hr := e_1261_17
  simp only [e_1261_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1261 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11617)) := by
  have c77619 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77620 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77621 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77622 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77623 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77624 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1261_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_18
  simp only [← c77619, ← c77620, ← c77621] at e_1261_18
  have e_1261_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_19
  simp only [← c77622, ← c77623, k2, ← c77624] at e_1261_19
  have hr := e_1261_19
  simp only [e_1261_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11618)) := by
  have c77625 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77626 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77627 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77628 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77629 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77630 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_0
  simp only [← c77625, ← c77626, ← c77627] at e_1262_0
  have e_1262_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_1
  simp only [← c77628, ← c77629, k2, ← c77630] at e_1262_1
  have hr := e_1262_1
  simp only [e_1262_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11619)) := by
  have c77631 := (publicBatchWrapper86_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77632 := (publicBatchWrapper86_copies2426 a h).1
  have c77633 := (publicBatchWrapper86_copies2426 a h).2.1
  have c77634 := (publicBatchWrapper86_copies2426 a h).2.2.1
  have c77635 := (publicBatchWrapper86_copies2426 a h).2.2.2.1
  have c77636 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_2
  simp only [← c77631, ← c77632, ← c77633] at e_1262_2
  have e_1262_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_3
  simp only [← c77634, ← c77635, k2, ← c77636] at e_1262_3
  have hr := e_1262_3
  simp only [e_1262_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11620)) := by
  have c77637 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.1
  have c77638 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.1
  have c77639 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.1
  have c77640 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.1
  have c77641 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.1
  have c77642 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_4
  simp only [← c77637, ← c77638, ← c77639] at e_1262_4
  have e_1262_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_5
  simp only [← c77640, ← c77641, k2, ← c77642] at e_1262_5
  have hr := e_1262_5
  simp only [e_1262_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11621)) := by
  have c77643 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77644 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77645 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77646 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77647 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77648 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_6
  simp only [← c77643, ← c77644, ← c77645] at e_1262_6
  have e_1262_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_7
  simp only [← c77646, ← c77647, k2, ← c77648] at e_1262_7
  have hr := e_1262_7
  simp only [e_1262_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11622)) := by
  have c77649 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77650 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77651 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77652 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77653 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77654 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_8
  simp only [← c77649, ← c77650, ← c77651] at e_1262_8
  have e_1262_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_9
  simp only [← c77652, ← c77653, k2, ← c77654] at e_1262_9
  have hr := e_1262_9
  simp only [e_1262_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11623)) := by
  have c77655 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77656 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77657 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77658 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77659 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77660 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_10
  simp only [← c77655, ← c77656, ← c77657] at e_1262_10
  have e_1262_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_11
  simp only [← c77658, ← c77659, k2, ← c77660] at e_1262_11
  have hr := e_1262_11
  simp only [e_1262_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11624)) := by
  have c77661 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77662 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77663 := (publicBatchWrapper86_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77664 := (publicBatchWrapper86_copies2427 a h).1
  have c77665 := (publicBatchWrapper86_copies2427 a h).2.1
  have c77666 := (publicBatchWrapper86_copies2427 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_12
  simp only [← c77661, ← c77662, ← c77663] at e_1262_12
  have e_1262_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_13
  simp only [← c77664, ← c77665, k2, ← c77666] at e_1262_13
  have hr := e_1262_13
  simp only [e_1262_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11625)) := by
  have c77667 := (publicBatchWrapper86_copies2427 a h).2.2.2.1
  have c77668 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.1
  have c77669 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.1
  have c77670 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.1
  have c77671 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.1
  have c77672 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_14
  simp only [← c77667, ← c77668, ← c77669] at e_1262_14
  have e_1262_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_15
  simp only [← c77670, ← c77671, k2, ← c77672] at e_1262_15
  have hr := e_1262_15
  simp only [e_1262_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11626)) := by
  have c77673 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.1
  have c77674 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77675 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77676 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77677 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77678 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_16
  simp only [← c77673, ← c77674, ← c77675] at e_1262_16
  have e_1262_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_17
  simp only [← c77676, ← c77677, k2, ← c77678] at e_1262_17
  have hr := e_1262_17
  simp only [e_1262_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1262 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11627)) := by
  have c77679 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77680 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77681 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77682 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77683 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77684 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1262_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_18
  simp only [← c77679, ← c77680, ← c77681] at e_1262_18
  have e_1262_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_19
  simp only [← c77682, ← c77683, k2, ← c77684] at e_1262_19
  have hr := e_1262_19
  simp only [e_1262_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11628)) := by
  have c77685 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77686 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77687 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77688 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77689 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77690 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_0
  simp only [← c77685, ← c77686, ← c77687] at e_1263_0
  have e_1263_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_1
  simp only [← c77688, ← c77689, k2, ← c77690] at e_1263_1
  have hr := e_1263_1
  simp only [e_1263_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11629)) := by
  have c77691 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77692 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77693 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77694 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77695 := (publicBatchWrapper86_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77696 := (publicBatchWrapper86_copies2428 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_2
  simp only [← c77691, ← c77692, ← c77693] at e_1263_2
  have e_1263_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_3
  simp only [← c77694, ← c77695, k2, ← c77696] at e_1263_3
  have hr := e_1263_3
  simp only [e_1263_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11630)) := by
  have c77697 := (publicBatchWrapper86_copies2428 a h).2.1
  have c77698 := (publicBatchWrapper86_copies2428 a h).2.2.1
  have c77699 := (publicBatchWrapper86_copies2428 a h).2.2.2.1
  have c77700 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.1
  have c77701 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.1
  have c77702 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_4
  simp only [← c77697, ← c77698, ← c77699] at e_1263_4
  have e_1263_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_5
  simp only [← c77700, ← c77701, k2, ← c77702] at e_1263_5
  have hr := e_1263_5
  simp only [e_1263_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11631)) := by
  have c77703 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.1
  have c77704 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.1
  have c77705 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.1
  have c77706 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77707 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77708 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_6
  simp only [← c77703, ← c77704, ← c77705] at e_1263_6
  have e_1263_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_7
  simp only [← c77706, ← c77707, k2, ← c77708] at e_1263_7
  have hr := e_1263_7
  simp only [e_1263_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11632)) := by
  have c77709 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77710 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77711 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77712 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77713 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77714 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_8
  simp only [← c77709, ← c77710, ← c77711] at e_1263_8
  have e_1263_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_9
  simp only [← c77712, ← c77713, k2, ← c77714] at e_1263_9
  have hr := e_1263_9
  simp only [e_1263_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11633)) := by
  have c77715 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77716 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77717 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77718 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77719 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77720 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_10
  simp only [← c77715, ← c77716, ← c77717] at e_1263_10
  have e_1263_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_11
  simp only [← c77718, ← c77719, k2, ← c77720] at e_1263_11
  have hr := e_1263_11
  simp only [e_1263_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11634)) := by
  have c77721 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77722 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77723 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77724 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77725 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77726 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_12
  simp only [← c77721, ← c77722, ← c77723] at e_1263_12
  have e_1263_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_13
  simp only [← c77724, ← c77725, k2, ← c77726] at e_1263_13
  have hr := e_1263_13
  simp only [e_1263_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11635)) := by
  have c77727 := (publicBatchWrapper86_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77728 := (publicBatchWrapper86_copies2429 a h).1
  have c77729 := (publicBatchWrapper86_copies2429 a h).2.1
  have c77730 := (publicBatchWrapper86_copies2429 a h).2.2.1
  have c77731 := (publicBatchWrapper86_copies2429 a h).2.2.2.1
  have c77732 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_14
  simp only [← c77727, ← c77728, ← c77729] at e_1263_14
  have e_1263_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_15
  simp only [← c77730, ← c77731, k2, ← c77732] at e_1263_15
  have hr := e_1263_15
  simp only [e_1263_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11636)) := by
  have c77733 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.1
  have c77734 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.1
  have c77735 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.1
  have c77736 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.1
  have c77737 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.1
  have c77738 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_16
  simp only [← c77733, ← c77734, ← c77735] at e_1263_16
  have e_1263_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_17
  simp only [← c77736, ← c77737, k2, ← c77738] at e_1263_17
  have hr := e_1263_17
  simp only [e_1263_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1263 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11637)) := by
  have c77739 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77740 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77741 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77742 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77743 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77744 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1263_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_18
  simp only [← c77739, ← c77740, ← c77741] at e_1263_18
  have e_1263_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_19
  simp only [← c77742, ← c77743, k2, ← c77744] at e_1263_19
  have hr := e_1263_19
  simp only [e_1263_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11638)) := by
  have c77745 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77746 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77747 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77748 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77749 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77750 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_0
  simp only [← c77745, ← c77746, ← c77747] at e_1264_0
  have e_1264_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_1
  simp only [← c77748, ← c77749, k2, ← c77750] at e_1264_1
  have hr := e_1264_1
  simp only [e_1264_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11639)) := by
  have c77751 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77752 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77753 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77754 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77755 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77756 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_2
  simp only [← c77751, ← c77752, ← c77753] at e_1264_2
  have e_1264_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_3
  simp only [← c77754, ← c77755, k2, ← c77756] at e_1264_3
  have hr := e_1264_3
  simp only [e_1264_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11640)) := by
  have c77757 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77758 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77759 := (publicBatchWrapper86_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77760 := (publicBatchWrapper86_copies2430 a h).1
  have c77761 := (publicBatchWrapper86_copies2430 a h).2.1
  have c77762 := (publicBatchWrapper86_copies2430 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_4
  simp only [← c77757, ← c77758, ← c77759] at e_1264_4
  have e_1264_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_5
  simp only [← c77760, ← c77761, k2, ← c77762] at e_1264_5
  have hr := e_1264_5
  simp only [e_1264_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11641)) := by
  have c77763 := (publicBatchWrapper86_copies2430 a h).2.2.2.1
  have c77764 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.1
  have c77765 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.1
  have c77766 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.1
  have c77767 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.1
  have c77768 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_6
  simp only [← c77763, ← c77764, ← c77765] at e_1264_6
  have e_1264_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_7
  simp only [← c77766, ← c77767, k2, ← c77768] at e_1264_7
  have hr := e_1264_7
  simp only [e_1264_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11642)) := by
  have c77769 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.1
  have c77770 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77771 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77772 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77773 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77774 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_8
  simp only [← c77769, ← c77770, ← c77771] at e_1264_8
  have e_1264_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_9
  simp only [← c77772, ← c77773, k2, ← c77774] at e_1264_9
  have hr := e_1264_9
  simp only [e_1264_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11643)) := by
  have c77775 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77776 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77777 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77778 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77779 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77780 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_10
  simp only [← c77775, ← c77776, ← c77777] at e_1264_10
  have e_1264_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_11
  simp only [← c77778, ← c77779, k2, ← c77780] at e_1264_11
  have hr := e_1264_11
  simp only [e_1264_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11644)) := by
  have c77781 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77782 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77783 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77784 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77785 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77786 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_12
  simp only [← c77781, ← c77782, ← c77783] at e_1264_12
  have e_1264_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_13
  simp only [← c77784, ← c77785, k2, ← c77786] at e_1264_13
  have hr := e_1264_13
  simp only [e_1264_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11645)) := by
  have c77787 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77788 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77789 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77790 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77791 := (publicBatchWrapper86_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77792 := (publicBatchWrapper86_copies2431 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_14
  simp only [← c77787, ← c77788, ← c77789] at e_1264_14
  have e_1264_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_15
  simp only [← c77790, ← c77791, k2, ← c77792] at e_1264_15
  have hr := e_1264_15
  simp only [e_1264_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11646)) := by
  have c77793 := (publicBatchWrapper86_copies2431 a h).2.1
  have c77794 := (publicBatchWrapper86_copies2431 a h).2.2.1
  have c77795 := (publicBatchWrapper86_copies2431 a h).2.2.2.1
  have c77796 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.1
  have c77797 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.1
  have c77798 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_16
  simp only [← c77793, ← c77794, ← c77795] at e_1264_16
  have e_1264_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_17
  simp only [← c77796, ← c77797, k2, ← c77798] at e_1264_17
  have hr := e_1264_17
  simp only [e_1264_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1264 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11647)) := by
  have c77799 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.1
  have c77800 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.1
  have c77801 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.1
  have c77802 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77803 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77804 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1264_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_18
  simp only [← c77799, ← c77800, ← c77801] at e_1264_18
  have e_1264_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_19
  simp only [← c77802, ← c77803, k2, ← c77804] at e_1264_19
  have hr := e_1264_19
  simp only [e_1264_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11648)) := by
  have c77805 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77806 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77807 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77808 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77809 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77810 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_0
  simp only [← c77805, ← c77806, ← c77807] at e_1265_0
  have e_1265_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_1
  simp only [← c77808, ← c77809, k2, ← c77810] at e_1265_1
  have hr := e_1265_1
  simp only [e_1265_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11649)) := by
  have c77811 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77812 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77813 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77814 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77815 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77816 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_2
  simp only [← c77811, ← c77812, ← c77813] at e_1265_2
  have e_1265_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_3
  simp only [← c77814, ← c77815, k2, ← c77816] at e_1265_3
  have hr := e_1265_3
  simp only [e_1265_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11650)) := by
  have c77817 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77818 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77819 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77820 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77821 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77822 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_4
  simp only [← c77817, ← c77818, ← c77819] at e_1265_4
  have e_1265_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_5
  simp only [← c77820, ← c77821, k2, ← c77822] at e_1265_5
  have hr := e_1265_5
  simp only [e_1265_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11651)) := by
  have c77823 := (publicBatchWrapper86_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77824 := (publicBatchWrapper86_copies2432 a h).1
  have c77825 := (publicBatchWrapper86_copies2432 a h).2.1
  have c77826 := (publicBatchWrapper86_copies2432 a h).2.2.1
  have c77827 := (publicBatchWrapper86_copies2432 a h).2.2.2.1
  have c77828 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_6
  simp only [← c77823, ← c77824, ← c77825] at e_1265_6
  have e_1265_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_7
  simp only [← c77826, ← c77827, k2, ← c77828] at e_1265_7
  have hr := e_1265_7
  simp only [e_1265_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11652)) := by
  have c77829 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.1
  have c77830 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.1
  have c77831 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.1
  have c77832 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.1
  have c77833 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.1
  have c77834 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_8
  simp only [← c77829, ← c77830, ← c77831] at e_1265_8
  have e_1265_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_9
  simp only [← c77832, ← c77833, k2, ← c77834] at e_1265_9
  have hr := e_1265_9
  simp only [e_1265_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11653)) := by
  have c77835 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77836 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77837 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77838 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77839 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77840 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_10
  simp only [← c77835, ← c77836, ← c77837] at e_1265_10
  have e_1265_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_11
  simp only [← c77838, ← c77839, k2, ← c77840] at e_1265_11
  have hr := e_1265_11
  simp only [e_1265_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11654)) := by
  have c77841 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77842 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77843 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77844 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77845 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77846 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_12
  simp only [← c77841, ← c77842, ← c77843] at e_1265_12
  have e_1265_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_13
  simp only [← c77844, ← c77845, k2, ← c77846] at e_1265_13
  have hr := e_1265_13
  simp only [e_1265_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11655)) := by
  have c77847 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77848 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77849 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77850 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77851 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77852 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_14
  simp only [← c77847, ← c77848, ← c77849] at e_1265_14
  have e_1265_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_15
  simp only [← c77850, ← c77851, k2, ← c77852] at e_1265_15
  have hr := e_1265_15
  simp only [e_1265_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11656)) := by
  have c77853 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77854 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77855 := (publicBatchWrapper86_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77856 := (publicBatchWrapper86_copies2433 a h).1
  have c77857 := (publicBatchWrapper86_copies2433 a h).2.1
  have c77858 := (publicBatchWrapper86_copies2433 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_16
  simp only [← c77853, ← c77854, ← c77855] at e_1265_16
  have e_1265_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_17
  simp only [← c77856, ← c77857, k2, ← c77858] at e_1265_17
  have hr := e_1265_17
  simp only [e_1265_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1265 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11657)) := by
  have c77859 := (publicBatchWrapper86_copies2433 a h).2.2.2.1
  have c77860 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.1
  have c77861 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.1
  have c77862 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.1
  have c77863 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.1
  have c77864 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1265_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_18
  simp only [← c77859, ← c77860, ← c77861] at e_1265_18
  have e_1265_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_19
  simp only [← c77862, ← c77863, k2, ← c77864] at e_1265_19
  have hr := e_1265_19
  simp only [e_1265_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11658)) := by
  have c77865 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.1
  have c77866 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77867 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77868 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77869 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77870 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_0
  simp only [← c77865, ← c77866, ← c77867] at e_1266_0
  have e_1266_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_1
  simp only [← c77868, ← c77869, k2, ← c77870] at e_1266_1
  have hr := e_1266_1
  simp only [e_1266_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11659)) := by
  have c77871 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77872 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77873 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77874 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77875 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77876 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_2
  simp only [← c77871, ← c77872, ← c77873] at e_1266_2
  have e_1266_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_3
  simp only [← c77874, ← c77875, k2, ← c77876] at e_1266_3
  have hr := e_1266_3
  simp only [e_1266_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11660)) := by
  have c77877 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77878 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77879 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77880 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77881 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77882 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_4
  simp only [← c77877, ← c77878, ← c77879] at e_1266_4
  have e_1266_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_5
  simp only [← c77880, ← c77881, k2, ← c77882] at e_1266_5
  have hr := e_1266_5
  simp only [e_1266_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11661)) := by
  have c77883 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77884 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77885 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77886 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77887 := (publicBatchWrapper86_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77888 := (publicBatchWrapper86_copies2434 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_6
  simp only [← c77883, ← c77884, ← c77885] at e_1266_6
  have e_1266_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_7
  simp only [← c77886, ← c77887, k2, ← c77888] at e_1266_7
  have hr := e_1266_7
  simp only [e_1266_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11662)) := by
  have c77889 := (publicBatchWrapper86_copies2434 a h).2.1
  have c77890 := (publicBatchWrapper86_copies2434 a h).2.2.1
  have c77891 := (publicBatchWrapper86_copies2434 a h).2.2.2.1
  have c77892 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.1
  have c77893 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.1
  have c77894 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_8
  simp only [← c77889, ← c77890, ← c77891] at e_1266_8
  have e_1266_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_9
  simp only [← c77892, ← c77893, k2, ← c77894] at e_1266_9
  have hr := e_1266_9
  simp only [e_1266_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11663)) := by
  have c77895 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.1
  have c77896 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.1
  have c77897 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.1
  have c77898 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77899 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77900 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_10
  simp only [← c77895, ← c77896, ← c77897] at e_1266_10
  have e_1266_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_11
  simp only [← c77898, ← c77899, k2, ← c77900] at e_1266_11
  have hr := e_1266_11
  simp only [e_1266_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11664)) := by
  have c77901 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77902 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77903 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77904 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77905 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77906 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_12
  simp only [← c77901, ← c77902, ← c77903] at e_1266_12
  have e_1266_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_13
  simp only [← c77904, ← c77905, k2, ← c77906] at e_1266_13
  have hr := e_1266_13
  simp only [e_1266_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11665)) := by
  have c77907 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77908 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77909 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77910 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77911 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77912 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_14
  simp only [← c77907, ← c77908, ← c77909] at e_1266_14
  have e_1266_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_15
  simp only [← c77910, ← c77911, k2, ← c77912] at e_1266_15
  have hr := e_1266_15
  simp only [e_1266_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11666)) := by
  have c77913 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77914 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77915 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77916 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77917 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77918 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_16
  simp only [← c77913, ← c77914, ← c77915] at e_1266_16
  have e_1266_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_17
  simp only [← c77916, ← c77917, k2, ← c77918] at e_1266_17
  have hr := e_1266_17
  simp only [e_1266_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1266 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11667)) := by
  have c77919 := (publicBatchWrapper86_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77920 := (publicBatchWrapper86_copies2435 a h).1
  have c77921 := (publicBatchWrapper86_copies2435 a h).2.1
  have c77922 := (publicBatchWrapper86_copies2435 a h).2.2.1
  have c77923 := (publicBatchWrapper86_copies2435 a h).2.2.2.1
  have c77924 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1266_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_18
  simp only [← c77919, ← c77920, ← c77921] at e_1266_18
  have e_1266_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_19
  simp only [← c77922, ← c77923, k2, ← c77924] at e_1266_19
  have hr := e_1266_19
  simp only [e_1266_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11668)) := by
  have c77925 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.1
  have c77926 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.1
  have c77927 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.1
  have c77928 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.1
  have c77929 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.1
  have c77930 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_0
  simp only [← c77925, ← c77926, ← c77927] at e_1267_0
  have e_1267_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_1
  simp only [← c77928, ← c77929, k2, ← c77930] at e_1267_1
  have hr := e_1267_1
  simp only [e_1267_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11669)) := by
  have c77931 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77932 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77933 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77934 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77935 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77936 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_2
  simp only [← c77931, ← c77932, ← c77933] at e_1267_2
  have e_1267_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_3
  simp only [← c77934, ← c77935, k2, ← c77936] at e_1267_3
  have hr := e_1267_3
  simp only [e_1267_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11670)) := by
  have c77937 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77938 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77939 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77940 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77941 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77942 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_4
  simp only [← c77937, ← c77938, ← c77939] at e_1267_4
  have e_1267_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_5
  simp only [← c77940, ← c77941, k2, ← c77942] at e_1267_5
  have hr := e_1267_5
  simp only [e_1267_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11671)) := by
  have c77943 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77944 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77945 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77946 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77947 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77948 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_6
  simp only [← c77943, ← c77944, ← c77945] at e_1267_6
  have e_1267_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_7
  simp only [← c77946, ← c77947, k2, ← c77948] at e_1267_7
  have hr := e_1267_7
  simp only [e_1267_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11672)) := by
  have c77949 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77950 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77951 := (publicBatchWrapper86_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77952 := (publicBatchWrapper86_copies2436 a h).1
  have c77953 := (publicBatchWrapper86_copies2436 a h).2.1
  have c77954 := (publicBatchWrapper86_copies2436 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_8
  simp only [← c77949, ← c77950, ← c77951] at e_1267_8
  have e_1267_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_9
  simp only [← c77952, ← c77953, k2, ← c77954] at e_1267_9
  have hr := e_1267_9
  simp only [e_1267_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11673)) := by
  have c77955 := (publicBatchWrapper86_copies2436 a h).2.2.2.1
  have c77956 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.1
  have c77957 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.1
  have c77958 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.1
  have c77959 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.1
  have c77960 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_10
  simp only [← c77955, ← c77956, ← c77957] at e_1267_10
  have e_1267_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_11
  simp only [← c77958, ← c77959, k2, ← c77960] at e_1267_11
  have hr := e_1267_11
  simp only [e_1267_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11674)) := by
  have c77961 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.1
  have c77962 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77963 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77964 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77965 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77966 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_12
  simp only [← c77961, ← c77962, ← c77963] at e_1267_12
  have e_1267_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_13
  simp only [← c77964, ← c77965, k2, ← c77966] at e_1267_13
  have hr := e_1267_13
  simp only [e_1267_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11675)) := by
  have c77967 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77968 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77969 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77970 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77971 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77972 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_14
  simp only [← c77967, ← c77968, ← c77969] at e_1267_14
  have e_1267_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_15
  simp only [← c77970, ← c77971, k2, ← c77972] at e_1267_15
  have hr := e_1267_15
  simp only [e_1267_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11676)) := by
  have c77973 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77974 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77975 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77976 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77977 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77978 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_16
  simp only [← c77973, ← c77974, ← c77975] at e_1267_16
  have e_1267_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_17
  simp only [← c77976, ← c77977, k2, ← c77978] at e_1267_17
  have hr := e_1267_17
  simp only [e_1267_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1267 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11677)) := by
  have c77979 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77980 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77981 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77982 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77983 := (publicBatchWrapper86_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77984 := (publicBatchWrapper86_copies2437 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1267_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_18
  simp only [← c77979, ← c77980, ← c77981] at e_1267_18
  have e_1267_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_19
  simp only [← c77982, ← c77983, k2, ← c77984] at e_1267_19
  have hr := e_1267_19
  simp only [e_1267_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11678)) := by
  have c77985 := (publicBatchWrapper86_copies2437 a h).2.1
  have c77986 := (publicBatchWrapper86_copies2437 a h).2.2.1
  have c77987 := (publicBatchWrapper86_copies2437 a h).2.2.2.1
  have c77988 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.1
  have c77989 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.1
  have c77990 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_0
  simp only [← c77985, ← c77986, ← c77987] at e_1268_0
  have e_1268_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_1
  simp only [← c77988, ← c77989, k2, ← c77990] at e_1268_1
  have hr := e_1268_1
  simp only [e_1268_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11679)) := by
  have c77991 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.1
  have c77992 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.1
  have c77993 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.1
  have c77994 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77995 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77996 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_2
  simp only [← c77991, ← c77992, ← c77993] at e_1268_2
  have e_1268_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_3
  simp only [← c77994, ← c77995, k2, ← c77996] at e_1268_3
  have hr := e_1268_3
  simp only [e_1268_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11680)) := by
  have c77997 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77998 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77999 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78000 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78001 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78002 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_4
  simp only [← c77997, ← c77998, ← c77999] at e_1268_4
  have e_1268_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_5
  simp only [← c78000, ← c78001, k2, ← c78002] at e_1268_5
  have hr := e_1268_5
  simp only [e_1268_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11681)) := by
  have c78003 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78004 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78005 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78006 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78007 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78008 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_6
  simp only [← c78003, ← c78004, ← c78005] at e_1268_6
  have e_1268_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_7
  simp only [← c78006, ← c78007, k2, ← c78008] at e_1268_7
  have hr := e_1268_7
  simp only [e_1268_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11682)) := by
  have c78009 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78010 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78011 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78012 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78013 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78014 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_8
  simp only [← c78009, ← c78010, ← c78011] at e_1268_8
  have e_1268_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_9
  simp only [← c78012, ← c78013, k2, ← c78014] at e_1268_9
  have hr := e_1268_9
  simp only [e_1268_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11683)) := by
  have c78015 := (publicBatchWrapper86_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78016 := (publicBatchWrapper86_copies2438 a h).1
  have c78017 := (publicBatchWrapper86_copies2438 a h).2.1
  have c78018 := (publicBatchWrapper86_copies2438 a h).2.2.1
  have c78019 := (publicBatchWrapper86_copies2438 a h).2.2.2.1
  have c78020 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_10
  simp only [← c78015, ← c78016, ← c78017] at e_1268_10
  have e_1268_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_11
  simp only [← c78018, ← c78019, k2, ← c78020] at e_1268_11
  have hr := e_1268_11
  simp only [e_1268_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11684)) := by
  have c78021 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.1
  have c78022 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.1
  have c78023 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.1
  have c78024 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.1
  have c78025 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.1
  have c78026 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_12
  simp only [← c78021, ← c78022, ← c78023] at e_1268_12
  have e_1268_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_13
  simp only [← c78024, ← c78025, k2, ← c78026] at e_1268_13
  have hr := e_1268_13
  simp only [e_1268_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11685)) := by
  have c78027 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78028 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78029 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78030 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78031 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78032 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_14
  simp only [← c78027, ← c78028, ← c78029] at e_1268_14
  have e_1268_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_15
  simp only [← c78030, ← c78031, k2, ← c78032] at e_1268_15
  have hr := e_1268_15
  simp only [e_1268_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11686)) := by
  have c78033 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78034 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78035 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78036 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78037 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78038 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_16
  simp only [← c78033, ← c78034, ← c78035] at e_1268_16
  have e_1268_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_17
  simp only [← c78036, ← c78037, k2, ← c78038] at e_1268_17
  have hr := e_1268_17
  simp only [e_1268_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1268 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11687)) := by
  have c78039 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78040 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78041 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78042 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78043 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78044 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1268_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_18
  simp only [← c78039, ← c78040, ← c78041] at e_1268_18
  have e_1268_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_19
  simp only [← c78042, ← c78043, k2, ← c78044] at e_1268_19
  have hr := e_1268_19
  simp only [e_1268_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11688)) := by
  have c78045 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78046 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78047 := (publicBatchWrapper86_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78048 := (publicBatchWrapper86_copies2439 a h).1
  have c78049 := (publicBatchWrapper86_copies2439 a h).2.1
  have c78050 := (publicBatchWrapper86_copies2439 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_0
  simp only [← c78045, ← c78046, ← c78047] at e_1269_0
  have e_1269_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_1
  simp only [← c78048, ← c78049, k2, ← c78050] at e_1269_1
  have hr := e_1269_1
  simp only [e_1269_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11689)) := by
  have c78051 := (publicBatchWrapper86_copies2439 a h).2.2.2.1
  have c78052 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.1
  have c78053 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.1
  have c78054 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.1
  have c78055 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.1
  have c78056 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_2
  simp only [← c78051, ← c78052, ← c78053] at e_1269_2
  have e_1269_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_3
  simp only [← c78054, ← c78055, k2, ← c78056] at e_1269_3
  have hr := e_1269_3
  simp only [e_1269_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11690)) := by
  have c78057 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.1
  have c78058 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78059 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78060 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78061 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78062 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_4
  simp only [← c78057, ← c78058, ← c78059] at e_1269_4
  have e_1269_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_5
  simp only [← c78060, ← c78061, k2, ← c78062] at e_1269_5
  have hr := e_1269_5
  simp only [e_1269_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11691)) := by
  have c78063 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78064 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78065 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78066 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78067 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78068 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_6
  simp only [← c78063, ← c78064, ← c78065] at e_1269_6
  have e_1269_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_7
  simp only [← c78066, ← c78067, k2, ← c78068] at e_1269_7
  have hr := e_1269_7
  simp only [e_1269_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11692)) := by
  have c78069 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78070 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78071 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78072 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78073 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78074 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_8
  simp only [← c78069, ← c78070, ← c78071] at e_1269_8
  have e_1269_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_9
  simp only [← c78072, ← c78073, k2, ← c78074] at e_1269_9
  have hr := e_1269_9
  simp only [e_1269_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11693)) := by
  have c78075 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78076 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78077 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78078 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78079 := (publicBatchWrapper86_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78080 := (publicBatchWrapper86_copies2440 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_10
  simp only [← c78075, ← c78076, ← c78077] at e_1269_10
  have e_1269_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_11
  simp only [← c78078, ← c78079, k2, ← c78080] at e_1269_11
  have hr := e_1269_11
  simp only [e_1269_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11694)) := by
  have c78081 := (publicBatchWrapper86_copies2440 a h).2.1
  have c78082 := (publicBatchWrapper86_copies2440 a h).2.2.1
  have c78083 := (publicBatchWrapper86_copies2440 a h).2.2.2.1
  have c78084 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.1
  have c78085 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.1
  have c78086 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_12
  simp only [← c78081, ← c78082, ← c78083] at e_1269_12
  have e_1269_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_13
  simp only [← c78084, ← c78085, k2, ← c78086] at e_1269_13
  have hr := e_1269_13
  simp only [e_1269_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11695)) := by
  have c78087 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.1
  have c78088 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.1
  have c78089 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.1
  have c78090 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78091 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78092 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_14
  simp only [← c78087, ← c78088, ← c78089] at e_1269_14
  have e_1269_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_15
  simp only [← c78090, ← c78091, k2, ← c78092] at e_1269_15
  have hr := e_1269_15
  simp only [e_1269_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11696)) := by
  have c78093 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78094 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78095 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78096 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78097 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78098 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_16
  simp only [← c78093, ← c78094, ← c78095] at e_1269_16
  have e_1269_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_17
  simp only [← c78096, ← c78097, k2, ← c78098] at e_1269_17
  have hr := e_1269_17
  simp only [e_1269_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1269 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11697)) := by
  have c78099 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78100 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78101 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78102 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78103 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78104 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1269_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_18
  simp only [← c78099, ← c78100, ← c78101] at e_1269_18
  have e_1269_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_19
  simp only [← c78102, ← c78103, k2, ← c78104] at e_1269_19
  have hr := e_1269_19
  simp only [e_1269_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11698)) := by
  have c78105 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78106 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78107 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78108 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78109 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78110 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_0
  simp only [← c78105, ← c78106, ← c78107] at e_1270_0
  have e_1270_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_1
  simp only [← c78108, ← c78109, k2, ← c78110] at e_1270_1
  have hr := e_1270_1
  simp only [e_1270_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11699)) := by
  have c78111 := (publicBatchWrapper86_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78112 := (publicBatchWrapper86_copies2441 a h).1
  have c78113 := (publicBatchWrapper86_copies2441 a h).2.1
  have c78114 := (publicBatchWrapper86_copies2441 a h).2.2.1
  have c78115 := (publicBatchWrapper86_copies2441 a h).2.2.2.1
  have c78116 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_2
  simp only [← c78111, ← c78112, ← c78113] at e_1270_2
  have e_1270_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_3
  simp only [← c78114, ← c78115, k2, ← c78116] at e_1270_3
  have hr := e_1270_3
  simp only [e_1270_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11700)) := by
  have c78117 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.1
  have c78118 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.1
  have c78119 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.1
  have c78120 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.1
  have c78121 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.1
  have c78122 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_4
  simp only [← c78117, ← c78118, ← c78119] at e_1270_4
  have e_1270_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_5
  simp only [← c78120, ← c78121, k2, ← c78122] at e_1270_5
  have hr := e_1270_5
  simp only [e_1270_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11701)) := by
  have c78123 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78124 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78125 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78126 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78127 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78128 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_6
  simp only [← c78123, ← c78124, ← c78125] at e_1270_6
  have e_1270_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_7
  simp only [← c78126, ← c78127, k2, ← c78128] at e_1270_7
  have hr := e_1270_7
  simp only [e_1270_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11702)) := by
  have c78129 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78130 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78131 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78132 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78133 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78134 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_8
  simp only [← c78129, ← c78130, ← c78131] at e_1270_8
  have e_1270_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_9
  simp only [← c78132, ← c78133, k2, ← c78134] at e_1270_9
  have hr := e_1270_9
  simp only [e_1270_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11703)) := by
  have c78135 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78136 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78137 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78138 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78139 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78140 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_10
  simp only [← c78135, ← c78136, ← c78137] at e_1270_10
  have e_1270_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_11
  simp only [← c78138, ← c78139, k2, ← c78140] at e_1270_11
  have hr := e_1270_11
  simp only [e_1270_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11704)) := by
  have c78141 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78142 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78143 := (publicBatchWrapper86_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78144 := (publicBatchWrapper86_copies2442 a h).1
  have c78145 := (publicBatchWrapper86_copies2442 a h).2.1
  have c78146 := (publicBatchWrapper86_copies2442 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_12
  simp only [← c78141, ← c78142, ← c78143] at e_1270_12
  have e_1270_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_13
  simp only [← c78144, ← c78145, k2, ← c78146] at e_1270_13
  have hr := e_1270_13
  simp only [e_1270_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11705)) := by
  have c78147 := (publicBatchWrapper86_copies2442 a h).2.2.2.1
  have c78148 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.1
  have c78149 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.1
  have c78150 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.1
  have c78151 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.1
  have c78152 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_14
  simp only [← c78147, ← c78148, ← c78149] at e_1270_14
  have e_1270_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_15
  simp only [← c78150, ← c78151, k2, ← c78152] at e_1270_15
  have hr := e_1270_15
  simp only [e_1270_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11706)) := by
  have c78153 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.1
  have c78154 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78155 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78156 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78157 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78158 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_16
  simp only [← c78153, ← c78154, ← c78155] at e_1270_16
  have e_1270_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_17
  simp only [← c78156, ← c78157, k2, ← c78158] at e_1270_17
  have hr := e_1270_17
  simp only [e_1270_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1270 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11707)) := by
  have c78159 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78160 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78161 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78162 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78163 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78164 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1270_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_18
  simp only [← c78159, ← c78160, ← c78161] at e_1270_18
  have e_1270_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_19
  simp only [← c78162, ← c78163, k2, ← c78164] at e_1270_19
  have hr := e_1270_19
  simp only [e_1270_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11708)) := by
  have c78165 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78166 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78167 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78168 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78169 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78170 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_0
  simp only [← c78165, ← c78166, ← c78167] at e_1271_0
  have e_1271_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_1
  simp only [← c78168, ← c78169, k2, ← c78170] at e_1271_1
  have hr := e_1271_1
  simp only [e_1271_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11709)) := by
  have c78171 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78172 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78173 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78174 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78175 := (publicBatchWrapper86_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78176 := (publicBatchWrapper86_copies2443 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_2
  simp only [← c78171, ← c78172, ← c78173] at e_1271_2
  have e_1271_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_3
  simp only [← c78174, ← c78175, k2, ← c78176] at e_1271_3
  have hr := e_1271_3
  simp only [e_1271_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11710)) := by
  have c78177 := (publicBatchWrapper86_copies2443 a h).2.1
  have c78178 := (publicBatchWrapper86_copies2443 a h).2.2.1
  have c78179 := (publicBatchWrapper86_copies2443 a h).2.2.2.1
  have c78180 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.1
  have c78181 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.1
  have c78182 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_4
  simp only [← c78177, ← c78178, ← c78179] at e_1271_4
  have e_1271_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_5
  simp only [← c78180, ← c78181, k2, ← c78182] at e_1271_5
  have hr := e_1271_5
  simp only [e_1271_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11711)) := by
  have c78183 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.1
  have c78184 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.1
  have c78185 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.1
  have c78186 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78187 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78188 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_6
  simp only [← c78183, ← c78184, ← c78185] at e_1271_6
  have e_1271_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_7
  simp only [← c78186, ← c78187, k2, ← c78188] at e_1271_7
  have hr := e_1271_7
  simp only [e_1271_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11712)) := by
  have c78189 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78190 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78191 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78192 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78193 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78194 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_8
  simp only [← c78189, ← c78190, ← c78191] at e_1271_8
  have e_1271_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_9
  simp only [← c78192, ← c78193, k2, ← c78194] at e_1271_9
  have hr := e_1271_9
  simp only [e_1271_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11713)) := by
  have c78195 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78196 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78197 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78198 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78199 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78200 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_10
  simp only [← c78195, ← c78196, ← c78197] at e_1271_10
  have e_1271_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_11
  simp only [← c78198, ← c78199, k2, ← c78200] at e_1271_11
  have hr := e_1271_11
  simp only [e_1271_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11714)) := by
  have c78201 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78202 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78203 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78204 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78205 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78206 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_12
  simp only [← c78201, ← c78202, ← c78203] at e_1271_12
  have e_1271_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_13
  simp only [← c78204, ← c78205, k2, ← c78206] at e_1271_13
  have hr := e_1271_13
  simp only [e_1271_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11715)) := by
  have c78207 := (publicBatchWrapper86_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78208 := (publicBatchWrapper86_copies2444 a h).1
  have c78209 := (publicBatchWrapper86_copies2444 a h).2.1
  have c78210 := (publicBatchWrapper86_copies2444 a h).2.2.1
  have c78211 := (publicBatchWrapper86_copies2444 a h).2.2.2.1
  have c78212 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_14
  simp only [← c78207, ← c78208, ← c78209] at e_1271_14
  have e_1271_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_15
  simp only [← c78210, ← c78211, k2, ← c78212] at e_1271_15
  have hr := e_1271_15
  simp only [e_1271_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11716)) := by
  have c78213 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.1
  have c78214 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.1
  have c78215 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.1
  have c78216 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.1
  have c78217 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.1
  have c78218 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_16
  simp only [← c78213, ← c78214, ← c78215] at e_1271_16
  have e_1271_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_17
  simp only [← c78216, ← c78217, k2, ← c78218] at e_1271_17
  have hr := e_1271_17
  simp only [e_1271_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1271 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11717)) := by
  have c78219 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78220 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78221 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78222 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78223 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78224 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1271_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_18
  simp only [← c78219, ← c78220, ← c78221] at e_1271_18
  have e_1271_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_19
  simp only [← c78222, ← c78223, k2, ← c78224] at e_1271_19
  have hr := e_1271_19
  simp only [e_1271_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11718)) := by
  have c78225 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78226 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78227 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78228 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78229 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78230 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_0
  simp only [← c78225, ← c78226, ← c78227] at e_1272_0
  have e_1272_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_1
  simp only [← c78228, ← c78229, k2, ← c78230] at e_1272_1
  have hr := e_1272_1
  simp only [e_1272_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11719)) := by
  have c78231 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78232 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78233 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78234 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78235 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78236 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_2
  simp only [← c78231, ← c78232, ← c78233] at e_1272_2
  have e_1272_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_3
  simp only [← c78234, ← c78235, k2, ← c78236] at e_1272_3
  have hr := e_1272_3
  simp only [e_1272_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11720)) := by
  have c78237 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78238 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78239 := (publicBatchWrapper86_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78240 := (publicBatchWrapper86_copies2445 a h).1
  have c78241 := (publicBatchWrapper86_copies2445 a h).2.1
  have c78242 := (publicBatchWrapper86_copies2445 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_4
  simp only [← c78237, ← c78238, ← c78239] at e_1272_4
  have e_1272_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_5
  simp only [← c78240, ← c78241, k2, ← c78242] at e_1272_5
  have hr := e_1272_5
  simp only [e_1272_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11721)) := by
  have c78243 := (publicBatchWrapper86_copies2445 a h).2.2.2.1
  have c78244 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.1
  have c78245 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.1
  have c78246 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.1
  have c78247 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.1
  have c78248 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_6
  simp only [← c78243, ← c78244, ← c78245] at e_1272_6
  have e_1272_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_7
  simp only [← c78246, ← c78247, k2, ← c78248] at e_1272_7
  have hr := e_1272_7
  simp only [e_1272_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11722)) := by
  have c78249 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.1
  have c78250 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78251 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78252 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78253 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78254 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_8
  simp only [← c78249, ← c78250, ← c78251] at e_1272_8
  have e_1272_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_9
  simp only [← c78252, ← c78253, k2, ← c78254] at e_1272_9
  have hr := e_1272_9
  simp only [e_1272_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11723)) := by
  have c78255 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78256 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78257 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78258 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78259 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78260 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_10
  simp only [← c78255, ← c78256, ← c78257] at e_1272_10
  have e_1272_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_11
  simp only [← c78258, ← c78259, k2, ← c78260] at e_1272_11
  have hr := e_1272_11
  simp only [e_1272_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11724)) := by
  have c78261 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78262 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78263 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78264 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78265 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78266 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_12
  simp only [← c78261, ← c78262, ← c78263] at e_1272_12
  have e_1272_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_13
  simp only [← c78264, ← c78265, k2, ← c78266] at e_1272_13
  have hr := e_1272_13
  simp only [e_1272_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11725)) := by
  have c78267 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78268 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78269 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78270 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78271 := (publicBatchWrapper86_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78272 := (publicBatchWrapper86_copies2446 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_14
  simp only [← c78267, ← c78268, ← c78269] at e_1272_14
  have e_1272_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_15
  simp only [← c78270, ← c78271, k2, ← c78272] at e_1272_15
  have hr := e_1272_15
  simp only [e_1272_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11726)) := by
  have c78273 := (publicBatchWrapper86_copies2446 a h).2.1
  have c78274 := (publicBatchWrapper86_copies2446 a h).2.2.1
  have c78275 := (publicBatchWrapper86_copies2446 a h).2.2.2.1
  have c78276 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.1
  have c78277 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.1
  have c78278 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_16
  simp only [← c78273, ← c78274, ← c78275] at e_1272_16
  have e_1272_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_17
  simp only [← c78276, ← c78277, k2, ← c78278] at e_1272_17
  have hr := e_1272_17
  simp only [e_1272_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11727)) := by
  have c78279 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.1
  have c78280 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.1
  have c78281 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.1
  have c78282 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78283 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78284 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_18
  simp only [← c78279, ← c78280, ← c78281] at e_1272_18
  have e_1272_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_19
  simp only [← c78282, ← c78283, k2, ← c78284] at e_1272_19
  have hr := e_1272_19
  simp only [e_1272_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11728)) := by
  have c78285 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78286 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78287 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78288 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78289 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78290 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_0
  simp only [← c78285, ← c78286, ← c78287] at e_1273_0
  have e_1273_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_1
  simp only [← c78288, ← c78289, k2, ← c78290] at e_1273_1
  have hr := e_1273_1
  simp only [e_1273_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11729)) := by
  have c78291 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78292 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78293 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78294 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78295 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78296 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_2
  simp only [← c78291, ← c78292, ← c78293] at e_1273_2
  have e_1273_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_3
  simp only [← c78294, ← c78295, k2, ← c78296] at e_1273_3
  have hr := e_1273_3
  simp only [e_1273_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11730)) := by
  have c78297 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78298 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78299 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78300 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78301 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78302 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_4
  simp only [← c78297, ← c78298, ← c78299] at e_1273_4
  have e_1273_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_5
  simp only [← c78300, ← c78301, k2, ← c78302] at e_1273_5
  have hr := e_1273_5
  simp only [e_1273_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11731)) := by
  have c78303 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78304 := (publicBatchWrapper86_copies2447 a h).1
  have c78305 := (publicBatchWrapper86_copies2447 a h).2.1
  have c78306 := (publicBatchWrapper86_copies2447 a h).2.2.1
  have c78307 := (publicBatchWrapper86_copies2447 a h).2.2.2.1
  have c78308 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_6
  simp only [← c78303, ← c78304, ← c78305] at e_1273_6
  have e_1273_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_7
  simp only [← c78306, ← c78307, k2, ← c78308] at e_1273_7
  have hr := e_1273_7
  simp only [e_1273_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11732)) := by
  have c78309 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.1
  have c78310 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.1
  have c78311 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.1
  have c78312 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.1
  have c78313 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.1
  have c78314 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_8
  simp only [← c78309, ← c78310, ← c78311] at e_1273_8
  have e_1273_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_9
  simp only [← c78312, ← c78313, k2, ← c78314] at e_1273_9
  have hr := e_1273_9
  simp only [e_1273_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11733)) := by
  have c78315 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78316 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78317 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78318 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78319 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78320 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_10
  simp only [← c78315, ← c78316, ← c78317] at e_1273_10
  have e_1273_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_11
  simp only [← c78318, ← c78319, k2, ← c78320] at e_1273_11
  have hr := e_1273_11
  simp only [e_1273_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11734)) := by
  have c78321 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78322 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78323 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78324 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78325 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78326 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_12
  simp only [← c78321, ← c78322, ← c78323] at e_1273_12
  have e_1273_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_13
  simp only [← c78324, ← c78325, k2, ← c78326] at e_1273_13
  have hr := e_1273_13
  simp only [e_1273_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11735)) := by
  have c78327 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78328 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78329 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78330 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78331 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78332 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_14
  simp only [← c78327, ← c78328, ← c78329] at e_1273_14
  have e_1273_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_15
  simp only [← c78330, ← c78331, k2, ← c78332] at e_1273_15
  have hr := e_1273_15
  simp only [e_1273_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11736)) := by
  have c78333 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78334 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78335 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78336 := (publicBatchWrapper86_copies2448 a h).1
  have c78337 := (publicBatchWrapper86_copies2448 a h).2.1
  have c78338 := (publicBatchWrapper86_copies2448 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_16
  simp only [← c78333, ← c78334, ← c78335] at e_1273_16
  have e_1273_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_17
  simp only [← c78336, ← c78337, k2, ← c78338] at e_1273_17
  have hr := e_1273_17
  simp only [e_1273_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11737)) := by
  have c78339 := (publicBatchWrapper86_copies2448 a h).2.2.2.1
  have c78340 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.1
  have c78341 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.1
  have c78342 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.1
  have c78343 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.1
  have c78344 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_18
  simp only [← c78339, ← c78340, ← c78341] at e_1273_18
  have e_1273_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_19
  simp only [← c78342, ← c78343, k2, ← c78344] at e_1273_19
  have hr := e_1273_19
  simp only [e_1273_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11738)) := by
  have c78345 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.1
  have c78346 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78347 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78348 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78349 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78350 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_0
  simp only [← c78345, ← c78346, ← c78347] at e_1274_0
  have e_1274_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_1
  simp only [← c78348, ← c78349, k2, ← c78350] at e_1274_1
  have hr := e_1274_1
  simp only [e_1274_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11739)) := by
  have c78351 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78352 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78353 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78354 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78355 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78356 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_2
  simp only [← c78351, ← c78352, ← c78353] at e_1274_2
  have e_1274_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_3
  simp only [← c78354, ← c78355, k2, ← c78356] at e_1274_3
  have hr := e_1274_3
  simp only [e_1274_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11740)) := by
  have c78357 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78358 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78359 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78360 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78361 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78362 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_4
  simp only [← c78357, ← c78358, ← c78359] at e_1274_4
  have e_1274_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_5
  simp only [← c78360, ← c78361, k2, ← c78362] at e_1274_5
  have hr := e_1274_5
  simp only [e_1274_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11741)) := by
  have c78363 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78364 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78365 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78366 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78367 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78368 := (publicBatchWrapper86_copies2449 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_6
  simp only [← c78363, ← c78364, ← c78365] at e_1274_6
  have e_1274_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_7
  simp only [← c78366, ← c78367, k2, ← c78368] at e_1274_7
  have hr := e_1274_7
  simp only [e_1274_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11742)) := by
  have c78369 := (publicBatchWrapper86_copies2449 a h).2.1
  have c78370 := (publicBatchWrapper86_copies2449 a h).2.2.1
  have c78371 := (publicBatchWrapper86_copies2449 a h).2.2.2.1
  have c78372 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.1
  have c78373 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.1
  have c78374 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_8
  simp only [← c78369, ← c78370, ← c78371] at e_1274_8
  have e_1274_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_9
  simp only [← c78372, ← c78373, k2, ← c78374] at e_1274_9
  have hr := e_1274_9
  simp only [e_1274_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11743)) := by
  have c78375 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.1
  have c78376 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.1
  have c78377 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.1
  have c78378 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78379 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78380 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_10
  simp only [← c78375, ← c78376, ← c78377] at e_1274_10
  have e_1274_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_11
  simp only [← c78378, ← c78379, k2, ← c78380] at e_1274_11
  have hr := e_1274_11
  simp only [e_1274_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11744)) := by
  have c78381 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78382 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78383 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78384 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78385 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78386 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_12
  simp only [← c78381, ← c78382, ← c78383] at e_1274_12
  have e_1274_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_13
  simp only [← c78384, ← c78385, k2, ← c78386] at e_1274_13
  have hr := e_1274_13
  simp only [e_1274_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11745)) := by
  have c78387 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78388 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78389 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78390 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78391 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78392 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_14
  simp only [← c78387, ← c78388, ← c78389] at e_1274_14
  have e_1274_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_15
  simp only [← c78390, ← c78391, k2, ← c78392] at e_1274_15
  have hr := e_1274_15
  simp only [e_1274_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11746)) := by
  have c78393 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78394 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78395 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78396 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78397 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78398 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_16
  simp only [← c78393, ← c78394, ← c78395] at e_1274_16
  have e_1274_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_17
  simp only [← c78396, ← c78397, k2, ← c78398] at e_1274_17
  have hr := e_1274_17
  simp only [e_1274_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11747)) := by
  have c78399 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78400 := (publicBatchWrapper86_copies2450 a h).1
  have c78401 := (publicBatchWrapper86_copies2450 a h).2.1
  have c78402 := (publicBatchWrapper86_copies2450 a h).2.2.1
  have c78403 := (publicBatchWrapper86_copies2450 a h).2.2.2.1
  have c78404 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_18
  simp only [← c78399, ← c78400, ← c78401] at e_1274_18
  have e_1274_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_19
  simp only [← c78402, ← c78403, k2, ← c78404] at e_1274_19
  have hr := e_1274_19
  simp only [e_1274_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11748)) := by
  have c78405 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.1
  have c78406 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.1
  have c78407 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.1
  have c78408 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.1
  have c78409 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.1
  have c78410 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_0
  simp only [← c78405, ← c78406, ← c78407] at e_1275_0
  have e_1275_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_1
  simp only [← c78408, ← c78409, k2, ← c78410] at e_1275_1
  have hr := e_1275_1
  simp only [e_1275_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11749)) := by
  have c78411 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78412 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78413 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78414 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78415 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78416 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_2
  simp only [← c78411, ← c78412, ← c78413] at e_1275_2
  have e_1275_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_3
  simp only [← c78414, ← c78415, k2, ← c78416] at e_1275_3
  have hr := e_1275_3
  simp only [e_1275_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11750)) := by
  have c78417 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78418 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78419 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78420 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78421 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78422 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_4
  simp only [← c78417, ← c78418, ← c78419] at e_1275_4
  have e_1275_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_5
  simp only [← c78420, ← c78421, k2, ← c78422] at e_1275_5
  have hr := e_1275_5
  simp only [e_1275_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11751)) := by
  have c78423 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78424 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78425 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78426 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78427 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78428 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_6
  simp only [← c78423, ← c78424, ← c78425] at e_1275_6
  have e_1275_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_7
  simp only [← c78426, ← c78427, k2, ← c78428] at e_1275_7
  have hr := e_1275_7
  simp only [e_1275_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11752)) := by
  have c78429 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78430 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78431 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78432 := (publicBatchWrapper86_copies2451 a h).1
  have c78433 := (publicBatchWrapper86_copies2451 a h).2.1
  have c78434 := (publicBatchWrapper86_copies2451 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_8
  simp only [← c78429, ← c78430, ← c78431] at e_1275_8
  have e_1275_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_9
  simp only [← c78432, ← c78433, k2, ← c78434] at e_1275_9
  have hr := e_1275_9
  simp only [e_1275_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11753)) := by
  have c78435 := (publicBatchWrapper86_copies2451 a h).2.2.2.1
  have c78436 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.1
  have c78437 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.1
  have c78438 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.1
  have c78439 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.1
  have c78440 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_10
  simp only [← c78435, ← c78436, ← c78437] at e_1275_10
  have e_1275_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_11
  simp only [← c78438, ← c78439, k2, ← c78440] at e_1275_11
  have hr := e_1275_11
  simp only [e_1275_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11754)) := by
  have c78441 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.1
  have c78442 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78443 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78444 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78445 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78446 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_12
  simp only [← c78441, ← c78442, ← c78443] at e_1275_12
  have e_1275_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_13
  simp only [← c78444, ← c78445, k2, ← c78446] at e_1275_13
  have hr := e_1275_13
  simp only [e_1275_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11755)) := by
  have c78447 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78448 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78449 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78450 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78451 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78452 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_14
  simp only [← c78447, ← c78448, ← c78449] at e_1275_14
  have e_1275_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_15
  simp only [← c78450, ← c78451, k2, ← c78452] at e_1275_15
  have hr := e_1275_15
  simp only [e_1275_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11756)) := by
  have c78453 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78454 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78455 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78456 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78457 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78458 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_16
  simp only [← c78453, ← c78454, ← c78455] at e_1275_16
  have e_1275_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_17
  simp only [← c78456, ← c78457, k2, ← c78458] at e_1275_17
  have hr := e_1275_17
  simp only [e_1275_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11757)) := by
  have c78459 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78460 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78461 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78462 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78463 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78464 := (publicBatchWrapper86_copies2452 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_18
  simp only [← c78459, ← c78460, ← c78461] at e_1275_18
  have e_1275_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_19
  simp only [← c78462, ← c78463, k2, ← c78464] at e_1275_19
  have hr := e_1275_19
  simp only [e_1275_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11758)) := by
  have c78465 := (publicBatchWrapper86_copies2452 a h).2.1
  have c78466 := (publicBatchWrapper86_copies2452 a h).2.2.1
  have c78467 := (publicBatchWrapper86_copies2452 a h).2.2.2.1
  have c78468 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.1
  have c78469 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.1
  have c78470 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_0
  simp only [← c78465, ← c78466, ← c78467] at e_1276_0
  have e_1276_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_1
  simp only [← c78468, ← c78469, k2, ← c78470] at e_1276_1
  have hr := e_1276_1
  simp only [e_1276_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11759)) := by
  have c78471 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.1
  have c78472 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.1
  have c78473 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.1
  have c78474 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78475 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78476 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_2
  simp only [← c78471, ← c78472, ← c78473] at e_1276_2
  have e_1276_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_3
  simp only [← c78474, ← c78475, k2, ← c78476] at e_1276_3
  have hr := e_1276_3
  simp only [e_1276_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11760)) := by
  have c78477 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78478 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78479 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78480 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78481 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78482 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_4
  simp only [← c78477, ← c78478, ← c78479] at e_1276_4
  have e_1276_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_5
  simp only [← c78480, ← c78481, k2, ← c78482] at e_1276_5
  have hr := e_1276_5
  simp only [e_1276_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11761)) := by
  have c78483 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78484 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78485 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78486 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78487 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78488 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_6
  simp only [← c78483, ← c78484, ← c78485] at e_1276_6
  have e_1276_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_7
  simp only [← c78486, ← c78487, k2, ← c78488] at e_1276_7
  have hr := e_1276_7
  simp only [e_1276_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11762)) := by
  have c78489 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78490 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78491 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78492 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78493 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78494 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_8
  simp only [← c78489, ← c78490, ← c78491] at e_1276_8
  have e_1276_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_9
  simp only [← c78492, ← c78493, k2, ← c78494] at e_1276_9
  have hr := e_1276_9
  simp only [e_1276_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11763)) := by
  have c78495 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78496 := (publicBatchWrapper86_copies2453 a h).1
  have c78497 := (publicBatchWrapper86_copies2453 a h).2.1
  have c78498 := (publicBatchWrapper86_copies2453 a h).2.2.1
  have c78499 := (publicBatchWrapper86_copies2453 a h).2.2.2.1
  have c78500 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_10
  simp only [← c78495, ← c78496, ← c78497] at e_1276_10
  have e_1276_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_11
  simp only [← c78498, ← c78499, k2, ← c78500] at e_1276_11
  have hr := e_1276_11
  simp only [e_1276_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11764)) := by
  have c78501 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.1
  have c78502 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.1
  have c78503 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.1
  have c78504 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.1
  have c78505 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.1
  have c78506 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_12
  simp only [← c78501, ← c78502, ← c78503] at e_1276_12
  have e_1276_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_13
  simp only [← c78504, ← c78505, k2, ← c78506] at e_1276_13
  have hr := e_1276_13
  simp only [e_1276_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11765)) := by
  have c78507 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78508 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78509 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78510 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78511 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78512 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_14
  simp only [← c78507, ← c78508, ← c78509] at e_1276_14
  have e_1276_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_15
  simp only [← c78510, ← c78511, k2, ← c78512] at e_1276_15
  have hr := e_1276_15
  simp only [e_1276_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11766)) := by
  have c78513 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78514 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78515 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78516 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78517 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78518 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_16
  simp only [← c78513, ← c78514, ← c78515] at e_1276_16
  have e_1276_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_17
  simp only [← c78516, ← c78517, k2, ← c78518] at e_1276_17
  have hr := e_1276_17
  simp only [e_1276_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11767)) := by
  have c78519 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78520 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78521 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78522 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78523 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78524 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_18
  simp only [← c78519, ← c78520, ← c78521] at e_1276_18
  have e_1276_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_19
  simp only [← c78522, ← c78523, k2, ← c78524] at e_1276_19
  have hr := e_1276_19
  simp only [e_1276_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11768)) := by
  have c78525 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78526 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78527 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78528 := (publicBatchWrapper86_copies2454 a h).1
  have c78529 := (publicBatchWrapper86_copies2454 a h).2.1
  have c78530 := (publicBatchWrapper86_copies2454 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_0
  simp only [← c78525, ← c78526, ← c78527] at e_1277_0
  have e_1277_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_1
  simp only [← c78528, ← c78529, k2, ← c78530] at e_1277_1
  have hr := e_1277_1
  simp only [e_1277_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11769)) := by
  have c78531 := (publicBatchWrapper86_copies2454 a h).2.2.2.1
  have c78532 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.1
  have c78533 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.1
  have c78534 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.1
  have c78535 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.1
  have c78536 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_2
  simp only [← c78531, ← c78532, ← c78533] at e_1277_2
  have e_1277_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_3
  simp only [← c78534, ← c78535, k2, ← c78536] at e_1277_3
  have hr := e_1277_3
  simp only [e_1277_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11810)) := by
  have c78537 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.1
  have c78538 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78539 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78540 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78541 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78542 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_4
  simp only [← c78537, ← c78538, ← c78539] at e_1277_4
  have e_1277_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_5
  simp only [← c78540, ← c78541, k2, ← c78542] at e_1277_5
  have hr := e_1277_5
  simp only [e_1277_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11811)) := by
  have c78543 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78544 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78545 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78546 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78547 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78548 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_6
  simp only [← c78543, ← c78544, ← c78545] at e_1277_6
  have e_1277_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_7
  simp only [← c78546, ← c78547, k2, ← c78548] at e_1277_7
  have hr := e_1277_7
  simp only [e_1277_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11812)) := by
  have c78549 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78550 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78551 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78552 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78553 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78554 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_8
  simp only [← c78549, ← c78550, ← c78551] at e_1277_8
  have e_1277_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_9
  simp only [← c78552, ← c78553, k2, ← c78554] at e_1277_9
  have hr := e_1277_9
  simp only [e_1277_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11813)) := by
  have c78555 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78556 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78557 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78558 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78559 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78560 := (publicBatchWrapper86_copies2455 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_10
  simp only [← c78555, ← c78556, ← c78557] at e_1277_10
  have e_1277_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_11
  simp only [← c78558, ← c78559, k2, ← c78560] at e_1277_11
  have hr := e_1277_11
  simp only [e_1277_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11814)) := by
  have c78561 := (publicBatchWrapper86_copies2455 a h).2.1
  have c78562 := (publicBatchWrapper86_copies2455 a h).2.2.1
  have c78563 := (publicBatchWrapper86_copies2455 a h).2.2.2.1
  have c78564 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.1
  have c78565 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.1
  have c78566 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_12
  simp only [← c78561, ← c78562, ← c78563] at e_1277_12
  have e_1277_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_13
  simp only [← c78564, ← c78565, k2, ← c78566] at e_1277_13
  have hr := e_1277_13
  simp only [e_1277_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11815)) := by
  have c78567 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.1
  have c78568 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.1
  have c78569 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.1
  have c78570 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78571 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78572 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_14
  simp only [← c78567, ← c78568, ← c78569] at e_1277_14
  have e_1277_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_15
  simp only [← c78570, ← c78571, k2, ← c78572] at e_1277_15
  have hr := e_1277_15
  simp only [e_1277_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11816)) := by
  have c78573 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78574 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78575 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78576 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78577 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78578 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_16
  simp only [← c78573, ← c78574, ← c78575] at e_1277_16
  have e_1277_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_17
  simp only [← c78576, ← c78577, k2, ← c78578] at e_1277_17
  have hr := e_1277_17
  simp only [e_1277_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11817)) := by
  have c78579 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78580 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78581 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78582 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78583 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78584 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_18
  simp only [← c78579, ← c78580, ← c78581] at e_1277_18
  have e_1277_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_19
  simp only [← c78582, ← c78583, k2, ← c78584] at e_1277_19
  have hr := e_1277_19
  simp only [e_1277_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11818)) := by
  have c78585 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78586 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78587 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78588 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78589 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78590 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_0
  simp only [← c78585, ← c78586, ← c78587] at e_1278_0
  have e_1278_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_1
  simp only [← c78588, ← c78589, k2, ← c78590] at e_1278_1
  have hr := e_1278_1
  simp only [e_1278_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11819)) := by
  have c78591 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78592 := (publicBatchWrapper86_copies2456 a h).1
  have c78593 := (publicBatchWrapper86_copies2456 a h).2.1
  have c78594 := (publicBatchWrapper86_copies2456 a h).2.2.1
  have c78595 := (publicBatchWrapper86_copies2456 a h).2.2.2.1
  have c78596 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_2
  simp only [← c78591, ← c78592, ← c78593] at e_1278_2
  have e_1278_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_3
  simp only [← c78594, ← c78595, k2, ← c78596] at e_1278_3
  have hr := e_1278_3
  simp only [e_1278_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11820)) := by
  have c78597 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.1
  have c78598 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.1
  have c78599 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.1
  have c78600 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.1
  have c78601 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.1
  have c78602 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_4
  simp only [← c78597, ← c78598, ← c78599] at e_1278_4
  have e_1278_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_5
  simp only [← c78600, ← c78601, k2, ← c78602] at e_1278_5
  have hr := e_1278_5
  simp only [e_1278_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11821)) := by
  have c78603 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78604 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78605 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78606 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78607 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78608 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_6
  simp only [← c78603, ← c78604, ← c78605] at e_1278_6
  have e_1278_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_7
  simp only [← c78606, ← c78607, k2, ← c78608] at e_1278_7
  have hr := e_1278_7
  simp only [e_1278_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11822)) := by
  have c78609 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78610 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78611 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78612 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78613 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78614 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_8
  simp only [← c78609, ← c78610, ← c78611] at e_1278_8
  have e_1278_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_9
  simp only [← c78612, ← c78613, k2, ← c78614] at e_1278_9
  have hr := e_1278_9
  simp only [e_1278_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11823)) := by
  have c78615 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78616 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78617 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78618 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78619 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78620 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_10
  simp only [← c78615, ← c78616, ← c78617] at e_1278_10
  have e_1278_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_11
  simp only [← c78618, ← c78619, k2, ← c78620] at e_1278_11
  have hr := e_1278_11
  simp only [e_1278_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11824)) := by
  have c78621 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78622 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78623 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78624 := (publicBatchWrapper86_copies2457 a h).1
  have c78625 := (publicBatchWrapper86_copies2457 a h).2.1
  have c78626 := (publicBatchWrapper86_copies2457 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_12
  simp only [← c78621, ← c78622, ← c78623] at e_1278_12
  have e_1278_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_13
  simp only [← c78624, ← c78625, k2, ← c78626] at e_1278_13
  have hr := e_1278_13
  simp only [e_1278_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11825)) := by
  have c78627 := (publicBatchWrapper86_copies2457 a h).2.2.2.1
  have c78628 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.1
  have c78629 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.1
  have c78630 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.1
  have c78631 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.1
  have c78632 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_14
  simp only [← c78627, ← c78628, ← c78629] at e_1278_14
  have e_1278_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_15
  simp only [← c78630, ← c78631, k2, ← c78632] at e_1278_15
  have hr := e_1278_15
  simp only [e_1278_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11826)) := by
  have c78633 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.1
  have c78634 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78635 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78636 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78637 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78638 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_16
  simp only [← c78633, ← c78634, ← c78635] at e_1278_16
  have e_1278_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_17
  simp only [← c78636, ← c78637, k2, ← c78638] at e_1278_17
  have hr := e_1278_17
  simp only [e_1278_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11827)) := by
  have c78639 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78640 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78641 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78642 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78643 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78644 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_18
  simp only [← c78639, ← c78640, ← c78641] at e_1278_18
  have e_1278_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_19
  simp only [← c78642, ← c78643, k2, ← c78644] at e_1278_19
  have hr := e_1278_19
  simp only [e_1278_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11828)) := by
  have c78645 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78646 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78647 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78648 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78649 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78650 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_0
  simp only [← c78645, ← c78646, ← c78647] at e_1279_0
  have e_1279_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_1
  simp only [← c78648, ← c78649, k2, ← c78650] at e_1279_1
  have hr := e_1279_1
  simp only [e_1279_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11829)) := by
  have c78651 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78652 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78653 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78654 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78655 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78656 := (publicBatchWrapper86_copies2458 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_2
  simp only [← c78651, ← c78652, ← c78653] at e_1279_2
  have e_1279_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_3
  simp only [← c78654, ← c78655, k2, ← c78656] at e_1279_3
  have hr := e_1279_3
  simp only [e_1279_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11830)) := by
  have c78657 := (publicBatchWrapper86_copies2458 a h).2.1
  have c78658 := (publicBatchWrapper86_copies2458 a h).2.2.1
  have c78659 := (publicBatchWrapper86_copies2458 a h).2.2.2.1
  have c78660 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.1
  have c78661 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.1
  have c78662 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_4
  simp only [← c78657, ← c78658, ← c78659] at e_1279_4
  have e_1279_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_5
  simp only [← c78660, ← c78661, k2, ← c78662] at e_1279_5
  have hr := e_1279_5
  simp only [e_1279_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11831)) := by
  have c78663 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.1
  have c78664 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.1
  have c78665 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.1
  have c78666 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78667 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78668 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_6
  simp only [← c78663, ← c78664, ← c78665] at e_1279_6
  have e_1279_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_7
  simp only [← c78666, ← c78667, k2, ← c78668] at e_1279_7
  have hr := e_1279_7
  simp only [e_1279_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11832)) := by
  have c78669 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78670 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78671 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78672 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78673 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78674 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_8
  simp only [← c78669, ← c78670, ← c78671] at e_1279_8
  have e_1279_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_9
  simp only [← c78672, ← c78673, k2, ← c78674] at e_1279_9
  have hr := e_1279_9
  simp only [e_1279_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11833)) := by
  have c78675 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78676 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78677 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78678 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78679 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78680 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_10
  simp only [← c78675, ← c78676, ← c78677] at e_1279_10
  have e_1279_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_11
  simp only [← c78678, ← c78679, k2, ← c78680] at e_1279_11
  have hr := e_1279_11
  simp only [e_1279_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11834)) := by
  have c78681 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78682 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78683 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78684 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78685 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78686 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_12
  simp only [← c78681, ← c78682, ← c78683] at e_1279_12
  have e_1279_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_13
  simp only [← c78684, ← c78685, k2, ← c78686] at e_1279_13
  have hr := e_1279_13
  simp only [e_1279_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11835)) := by
  have c78687 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78688 := (publicBatchWrapper86_copies2459 a h).1
  have c78689 := (publicBatchWrapper86_copies2459 a h).2.1
  have c78690 := (publicBatchWrapper86_copies2459 a h).2.2.1
  have c78691 := (publicBatchWrapper86_copies2459 a h).2.2.2.1
  have c78692 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_14
  simp only [← c78687, ← c78688, ← c78689] at e_1279_14
  have e_1279_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_15
  simp only [← c78690, ← c78691, k2, ← c78692] at e_1279_15
  have hr := e_1279_15
  simp only [e_1279_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11836)) := by
  have c78693 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.1
  have c78694 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.1
  have c78695 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.1
  have c78696 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.1
  have c78697 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.1
  have c78698 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_16
  simp only [← c78693, ← c78694, ← c78695] at e_1279_16
  have e_1279_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_17
  simp only [← c78696, ← c78697, k2, ← c78698] at e_1279_17
  have hr := e_1279_17
  simp only [e_1279_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11837)) := by
  have c78699 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78700 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78701 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78702 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78703 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78704 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_18
  simp only [← c78699, ← c78700, ← c78701] at e_1279_18
  have e_1279_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_19
  simp only [← c78702, ← c78703, k2, ← c78704] at e_1279_19
  have hr := e_1279_19
  simp only [e_1279_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11838)) := by
  have c78705 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78706 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78707 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78708 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78709 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78710 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_0
  simp only [← c78705, ← c78706, ← c78707] at e_1280_0
  have e_1280_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_1
  simp only [← c78708, ← c78709, k2, ← c78710] at e_1280_1
  have hr := e_1280_1
  simp only [e_1280_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11839)) := by
  have c78711 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78712 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78713 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78714 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78715 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78716 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_2
  simp only [← c78711, ← c78712, ← c78713] at e_1280_2
  have e_1280_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_3
  simp only [← c78714, ← c78715, k2, ← c78716] at e_1280_3
  have hr := e_1280_3
  simp only [e_1280_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11840)) := by
  have c78717 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78718 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78719 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78720 := (publicBatchWrapper86_copies2460 a h).1
  have c78721 := (publicBatchWrapper86_copies2460 a h).2.1
  have c78722 := (publicBatchWrapper86_copies2460 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_4
  simp only [← c78717, ← c78718, ← c78719] at e_1280_4
  have e_1280_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_5
  simp only [← c78720, ← c78721, k2, ← c78722] at e_1280_5
  have hr := e_1280_5
  simp only [e_1280_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11841)) := by
  have c78723 := (publicBatchWrapper86_copies2460 a h).2.2.2.1
  have c78724 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.1
  have c78725 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.1
  have c78726 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.1
  have c78727 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.1
  have c78728 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_6
  simp only [← c78723, ← c78724, ← c78725] at e_1280_6
  have e_1280_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_7
  simp only [← c78726, ← c78727, k2, ← c78728] at e_1280_7
  have hr := e_1280_7
  simp only [e_1280_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11842)) := by
  have c78729 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.1
  have c78730 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78731 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78732 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78733 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78734 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_8
  simp only [← c78729, ← c78730, ← c78731] at e_1280_8
  have e_1280_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_9
  simp only [← c78732, ← c78733, k2, ← c78734] at e_1280_9
  have hr := e_1280_9
  simp only [e_1280_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11843)) := by
  have c78735 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78736 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78737 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78738 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78739 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78740 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_10
  simp only [← c78735, ← c78736, ← c78737] at e_1280_10
  have e_1280_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_11
  simp only [← c78738, ← c78739, k2, ← c78740] at e_1280_11
  have hr := e_1280_11
  simp only [e_1280_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11844)) := by
  have c78741 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78742 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78743 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78744 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78745 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78746 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_12
  simp only [← c78741, ← c78742, ← c78743] at e_1280_12
  have e_1280_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_13
  simp only [← c78744, ← c78745, k2, ← c78746] at e_1280_13
  have hr := e_1280_13
  simp only [e_1280_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11845)) := by
  have c78747 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78748 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78749 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78750 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78751 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78752 := (publicBatchWrapper86_copies2461 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_14
  simp only [← c78747, ← c78748, ← c78749] at e_1280_14
  have e_1280_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_15
  simp only [← c78750, ← c78751, k2, ← c78752] at e_1280_15
  have hr := e_1280_15
  simp only [e_1280_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11846)) := by
  have c78753 := (publicBatchWrapper86_copies2461 a h).2.1
  have c78754 := (publicBatchWrapper86_copies2461 a h).2.2.1
  have c78755 := (publicBatchWrapper86_copies2461 a h).2.2.2.1
  have c78756 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.1
  have c78757 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.1
  have c78758 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_16
  simp only [← c78753, ← c78754, ← c78755] at e_1280_16
  have e_1280_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_17
  simp only [← c78756, ← c78757, k2, ← c78758] at e_1280_17
  have hr := e_1280_17
  simp only [e_1280_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11847)) := by
  have c78759 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.1
  have c78760 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.1
  have c78761 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.1
  have c78762 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78763 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78764 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_18
  simp only [← c78759, ← c78760, ← c78761] at e_1280_18
  have e_1280_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_19
  simp only [← c78762, ← c78763, k2, ← c78764] at e_1280_19
  have hr := e_1280_19
  simp only [e_1280_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11848)) := by
  have c78765 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78766 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78767 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78768 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78769 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78770 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_0
  simp only [← c78765, ← c78766, ← c78767] at e_1281_0
  have e_1281_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_1
  simp only [← c78768, ← c78769, k2, ← c78770] at e_1281_1
  have hr := e_1281_1
  simp only [e_1281_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11849)) := by
  have c78771 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78772 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78773 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78774 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78775 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78776 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_2
  simp only [← c78771, ← c78772, ← c78773] at e_1281_2
  have e_1281_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_3
  simp only [← c78774, ← c78775, k2, ← c78776] at e_1281_3
  have hr := e_1281_3
  simp only [e_1281_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11850)) := by
  have c78777 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78778 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78779 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78780 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78781 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78782 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_4
  simp only [← c78777, ← c78778, ← c78779] at e_1281_4
  have e_1281_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_5
  simp only [← c78780, ← c78781, k2, ← c78782] at e_1281_5
  have hr := e_1281_5
  simp only [e_1281_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11851)) := by
  have c78783 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78784 := (publicBatchWrapper86_copies2462 a h).1
  have c78785 := (publicBatchWrapper86_copies2462 a h).2.1
  have c78786 := (publicBatchWrapper86_copies2462 a h).2.2.1
  have c78787 := (publicBatchWrapper86_copies2462 a h).2.2.2.1
  have c78788 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_6
  simp only [← c78783, ← c78784, ← c78785] at e_1281_6
  have e_1281_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_7
  simp only [← c78786, ← c78787, k2, ← c78788] at e_1281_7
  have hr := e_1281_7
  simp only [e_1281_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11852)) := by
  have c78789 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.1
  have c78790 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.1
  have c78791 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.1
  have c78792 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.1
  have c78793 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.1
  have c78794 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_8
  simp only [← c78789, ← c78790, ← c78791] at e_1281_8
  have e_1281_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_9
  simp only [← c78792, ← c78793, k2, ← c78794] at e_1281_9
  have hr := e_1281_9
  simp only [e_1281_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11853)) := by
  have c78795 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78796 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78797 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78798 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78799 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78800 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_10
  simp only [← c78795, ← c78796, ← c78797] at e_1281_10
  have e_1281_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_11
  simp only [← c78798, ← c78799, k2, ← c78800] at e_1281_11
  have hr := e_1281_11
  simp only [e_1281_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11854)) := by
  have c78801 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78802 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78803 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78804 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78805 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78806 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_12
  simp only [← c78801, ← c78802, ← c78803] at e_1281_12
  have e_1281_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_13
  simp only [← c78804, ← c78805, k2, ← c78806] at e_1281_13
  have hr := e_1281_13
  simp only [e_1281_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11855)) := by
  have c78807 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78808 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78809 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78810 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78811 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78812 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_14
  simp only [← c78807, ← c78808, ← c78809] at e_1281_14
  have e_1281_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_15
  simp only [← c78810, ← c78811, k2, ← c78812] at e_1281_15
  have hr := e_1281_15
  simp only [e_1281_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11856)) := by
  have c78813 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78814 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78815 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78816 := (publicBatchWrapper86_copies2463 a h).1
  have c78817 := (publicBatchWrapper86_copies2463 a h).2.1
  have c78818 := (publicBatchWrapper86_copies2463 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_16
  simp only [← c78813, ← c78814, ← c78815] at e_1281_16
  have e_1281_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_17
  simp only [← c78816, ← c78817, k2, ← c78818] at e_1281_17
  have hr := e_1281_17
  simp only [e_1281_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11857)) := by
  have c78819 := (publicBatchWrapper86_copies2463 a h).2.2.2.1
  have c78820 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.1
  have c78821 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.1
  have c78822 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.1
  have c78823 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.1
  have c78824 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_18
  simp only [← c78819, ← c78820, ← c78821] at e_1281_18
  have e_1281_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_19
  simp only [← c78822, ← c78823, k2, ← c78824] at e_1281_19
  have hr := e_1281_19
  simp only [e_1281_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11858)) := by
  have c78825 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.1
  have c78826 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78827 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78828 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78829 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78830 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_0
  simp only [← c78825, ← c78826, ← c78827] at e_1282_0
  have e_1282_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_1
  simp only [← c78828, ← c78829, k2, ← c78830] at e_1282_1
  have hr := e_1282_1
  simp only [e_1282_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11859)) := by
  have c78831 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78832 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78833 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78834 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78835 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78836 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_2
  simp only [← c78831, ← c78832, ← c78833] at e_1282_2
  have e_1282_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_3
  simp only [← c78834, ← c78835, k2, ← c78836] at e_1282_3
  have hr := e_1282_3
  simp only [e_1282_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11860)) := by
  have c78837 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78838 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78839 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78840 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78841 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78842 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_4
  simp only [← c78837, ← c78838, ← c78839] at e_1282_4
  have e_1282_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_5
  simp only [← c78840, ← c78841, k2, ← c78842] at e_1282_5
  have hr := e_1282_5
  simp only [e_1282_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11861)) := by
  have c78843 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78844 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78845 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78846 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78847 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78848 := (publicBatchWrapper86_copies2464 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_6
  simp only [← c78843, ← c78844, ← c78845] at e_1282_6
  have e_1282_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_7
  simp only [← c78846, ← c78847, k2, ← c78848] at e_1282_7
  have hr := e_1282_7
  simp only [e_1282_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11862)) := by
  have c78849 := (publicBatchWrapper86_copies2464 a h).2.1
  have c78850 := (publicBatchWrapper86_copies2464 a h).2.2.1
  have c78851 := (publicBatchWrapper86_copies2464 a h).2.2.2.1
  have c78852 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.1
  have c78853 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.1
  have c78854 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_8
  simp only [← c78849, ← c78850, ← c78851] at e_1282_8
  have e_1282_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_9
  simp only [← c78852, ← c78853, k2, ← c78854] at e_1282_9
  have hr := e_1282_9
  simp only [e_1282_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11863)) := by
  have c78855 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.1
  have c78856 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.1
  have c78857 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.1
  have c78858 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78859 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78860 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_10
  simp only [← c78855, ← c78856, ← c78857] at e_1282_10
  have e_1282_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_11
  simp only [← c78858, ← c78859, k2, ← c78860] at e_1282_11
  have hr := e_1282_11
  simp only [e_1282_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11864)) := by
  have c78861 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78862 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78863 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78864 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78865 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78866 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_12
  simp only [← c78861, ← c78862, ← c78863] at e_1282_12
  have e_1282_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_13
  simp only [← c78864, ← c78865, k2, ← c78866] at e_1282_13
  have hr := e_1282_13
  simp only [e_1282_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11865)) := by
  have c78867 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78868 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78869 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78870 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78871 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78872 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_14
  simp only [← c78867, ← c78868, ← c78869] at e_1282_14
  have e_1282_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_15
  simp only [← c78870, ← c78871, k2, ← c78872] at e_1282_15
  have hr := e_1282_15
  simp only [e_1282_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11866)) := by
  have c78873 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78874 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78875 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78876 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78877 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78878 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_16
  simp only [← c78873, ← c78874, ← c78875] at e_1282_16
  have e_1282_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_17
  simp only [← c78876, ← c78877, k2, ← c78878] at e_1282_17
  have hr := e_1282_17
  simp only [e_1282_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11867)) := by
  have c78879 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78880 := (publicBatchWrapper86_copies2465 a h).1
  have c78881 := (publicBatchWrapper86_copies2465 a h).2.1
  have c78882 := (publicBatchWrapper86_copies2465 a h).2.2.1
  have c78883 := (publicBatchWrapper86_copies2465 a h).2.2.2.1
  have c78884 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_18
  simp only [← c78879, ← c78880, ← c78881] at e_1282_18
  have e_1282_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_19
  simp only [← c78882, ← c78883, k2, ← c78884] at e_1282_19
  have hr := e_1282_19
  simp only [e_1282_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11868)) := by
  have c78885 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.1
  have c78886 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.1
  have c78887 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.1
  have c78888 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.1
  have c78889 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.1
  have c78890 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_0
  simp only [← c78885, ← c78886, ← c78887] at e_1283_0
  have e_1283_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_1
  simp only [← c78888, ← c78889, k2, ← c78890] at e_1283_1
  have hr := e_1283_1
  simp only [e_1283_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11869)) := by
  have c78891 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78892 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78893 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78894 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78895 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78896 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_2
  simp only [← c78891, ← c78892, ← c78893] at e_1283_2
  have e_1283_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_3
  simp only [← c78894, ← c78895, k2, ← c78896] at e_1283_3
  have hr := e_1283_3
  simp only [e_1283_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11870)) := by
  have c78897 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78898 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78899 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78900 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78901 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78902 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_4
  simp only [← c78897, ← c78898, ← c78899] at e_1283_4
  have e_1283_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_5
  simp only [← c78900, ← c78901, k2, ← c78902] at e_1283_5
  have hr := e_1283_5
  simp only [e_1283_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11871)) := by
  have c78903 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78904 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78905 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78906 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78907 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78908 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_6
  simp only [← c78903, ← c78904, ← c78905] at e_1283_6
  have e_1283_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_7
  simp only [← c78906, ← c78907, k2, ← c78908] at e_1283_7
  have hr := e_1283_7
  simp only [e_1283_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11872)) := by
  have c78909 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78910 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78911 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78912 := (publicBatchWrapper86_copies2466 a h).1
  have c78913 := (publicBatchWrapper86_copies2466 a h).2.1
  have c78914 := (publicBatchWrapper86_copies2466 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_8
  simp only [← c78909, ← c78910, ← c78911] at e_1283_8
  have e_1283_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_9
  simp only [← c78912, ← c78913, k2, ← c78914] at e_1283_9
  have hr := e_1283_9
  simp only [e_1283_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11873)) := by
  have c78915 := (publicBatchWrapper86_copies2466 a h).2.2.2.1
  have c78916 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.1
  have c78917 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.1
  have c78918 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.1
  have c78919 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.1
  have c78920 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_10
  simp only [← c78915, ← c78916, ← c78917] at e_1283_10
  have e_1283_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_11
  simp only [← c78918, ← c78919, k2, ← c78920] at e_1283_11
  have hr := e_1283_11
  simp only [e_1283_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11874)) := by
  have c78921 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.1
  have c78922 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78923 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78924 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78925 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78926 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_12
  simp only [← c78921, ← c78922, ← c78923] at e_1283_12
  have e_1283_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_13
  simp only [← c78924, ← c78925, k2, ← c78926] at e_1283_13
  have hr := e_1283_13
  simp only [e_1283_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11875)) := by
  have c78927 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78928 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78929 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78930 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78931 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78932 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_14
  simp only [← c78927, ← c78928, ← c78929] at e_1283_14
  have e_1283_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_15
  simp only [← c78930, ← c78931, k2, ← c78932] at e_1283_15
  have hr := e_1283_15
  simp only [e_1283_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11876)) := by
  have c78933 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78934 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78935 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78936 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78937 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78938 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_16
  simp only [← c78933, ← c78934, ← c78935] at e_1283_16
  have e_1283_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_17
  simp only [← c78936, ← c78937, k2, ← c78938] at e_1283_17
  have hr := e_1283_17
  simp only [e_1283_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11877)) := by
  have c78939 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78940 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78941 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78942 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78943 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78944 := (publicBatchWrapper86_copies2467 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_18
  simp only [← c78939, ← c78940, ← c78941] at e_1283_18
  have e_1283_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_19
  simp only [← c78942, ← c78943, k2, ← c78944] at e_1283_19
  have hr := e_1283_19
  simp only [e_1283_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11878)) := by
  have c78945 := (publicBatchWrapper86_copies2467 a h).2.1
  have c78946 := (publicBatchWrapper86_copies2467 a h).2.2.1
  have c78947 := (publicBatchWrapper86_copies2467 a h).2.2.2.1
  have c78948 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.1
  have c78949 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.1
  have c78950 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_0
  simp only [← c78945, ← c78946, ← c78947] at e_1284_0
  have e_1284_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_1
  simp only [← c78948, ← c78949, k2, ← c78950] at e_1284_1
  have hr := e_1284_1
  simp only [e_1284_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11879)) := by
  have c78951 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.1
  have c78952 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.1
  have c78953 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.1
  have c78954 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78955 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78956 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_2
  simp only [← c78951, ← c78952, ← c78953] at e_1284_2
  have e_1284_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_3
  simp only [← c78954, ← c78955, k2, ← c78956] at e_1284_3
  have hr := e_1284_3
  simp only [e_1284_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11880)) := by
  have c78957 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78958 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78959 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78960 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78961 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78962 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_4
  simp only [← c78957, ← c78958, ← c78959] at e_1284_4
  have e_1284_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_5
  simp only [← c78960, ← c78961, k2, ← c78962] at e_1284_5
  have hr := e_1284_5
  simp only [e_1284_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11881)) := by
  have c78963 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78964 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78965 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78966 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78967 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78968 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_6
  simp only [← c78963, ← c78964, ← c78965] at e_1284_6
  have e_1284_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_7
  simp only [← c78966, ← c78967, k2, ← c78968] at e_1284_7
  have hr := e_1284_7
  simp only [e_1284_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11882)) := by
  have c78969 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78970 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78971 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78972 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78973 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78974 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_8
  simp only [← c78969, ← c78970, ← c78971] at e_1284_8
  have e_1284_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_9
  simp only [← c78972, ← c78973, k2, ← c78974] at e_1284_9
  have hr := e_1284_9
  simp only [e_1284_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11883)) := by
  have c78975 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78976 := (publicBatchWrapper86_copies2468 a h).1
  have c78977 := (publicBatchWrapper86_copies2468 a h).2.1
  have c78978 := (publicBatchWrapper86_copies2468 a h).2.2.1
  have c78979 := (publicBatchWrapper86_copies2468 a h).2.2.2.1
  have c78980 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_10
  simp only [← c78975, ← c78976, ← c78977] at e_1284_10
  have e_1284_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_11
  simp only [← c78978, ← c78979, k2, ← c78980] at e_1284_11
  have hr := e_1284_11
  simp only [e_1284_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11884)) := by
  have c78981 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.1
  have c78982 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.1
  have c78983 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.1
  have c78984 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.1
  have c78985 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.1
  have c78986 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_12
  simp only [← c78981, ← c78982, ← c78983] at e_1284_12
  have e_1284_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_13
  simp only [← c78984, ← c78985, k2, ← c78986] at e_1284_13
  have hr := e_1284_13
  simp only [e_1284_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11885)) := by
  have c78987 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78988 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78989 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78990 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78991 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78992 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_14
  simp only [← c78987, ← c78988, ← c78989] at e_1284_14
  have e_1284_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_15
  simp only [← c78990, ← c78991, k2, ← c78992] at e_1284_15
  have hr := e_1284_15
  simp only [e_1284_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11886)) := by
  have c78993 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78994 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78995 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78996 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78997 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78998 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_16
  simp only [← c78993, ← c78994, ← c78995] at e_1284_16
  have e_1284_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_17
  simp only [← c78996, ← c78997, k2, ← c78998] at e_1284_17
  have hr := e_1284_17
  simp only [e_1284_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11887)) := by
  have c78999 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79000 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79001 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79002 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79003 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79004 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_18
  simp only [← c78999, ← c79000, ← c79001] at e_1284_18
  have e_1284_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_19
  simp only [← c79002, ← c79003, k2, ← c79004] at e_1284_19
  have hr := e_1284_19
  simp only [e_1284_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11888)) := by
  have c79005 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79006 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79007 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79008 := (publicBatchWrapper86_copies2469 a h).1
  have c79009 := (publicBatchWrapper86_copies2469 a h).2.1
  have c79010 := (publicBatchWrapper86_copies2469 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_0
  simp only [← c79005, ← c79006, ← c79007] at e_1285_0
  have e_1285_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_1
  simp only [← c79008, ← c79009, k2, ← c79010] at e_1285_1
  have hr := e_1285_1
  simp only [e_1285_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11889)) := by
  have c79011 := (publicBatchWrapper86_copies2469 a h).2.2.2.1
  have c79012 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.1
  have c79013 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.1
  have c79014 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.1
  have c79015 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.1
  have c79016 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_2
  simp only [← c79011, ← c79012, ← c79013] at e_1285_2
  have e_1285_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_3
  simp only [← c79014, ← c79015, k2, ← c79016] at e_1285_3
  have hr := e_1285_3
  simp only [e_1285_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11890)) := by
  have c79017 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.1
  have c79018 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79019 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79020 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79021 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79022 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_4
  simp only [← c79017, ← c79018, ← c79019] at e_1285_4
  have e_1285_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_5
  simp only [← c79020, ← c79021, k2, ← c79022] at e_1285_5
  have hr := e_1285_5
  simp only [e_1285_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11891)) := by
  have c79023 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79024 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79025 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79026 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79027 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79028 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_6
  simp only [← c79023, ← c79024, ← c79025] at e_1285_6
  have e_1285_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_7
  simp only [← c79026, ← c79027, k2, ← c79028] at e_1285_7
  have hr := e_1285_7
  simp only [e_1285_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11892)) := by
  have c79029 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79030 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79031 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79032 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79033 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79034 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_8
  simp only [← c79029, ← c79030, ← c79031] at e_1285_8
  have e_1285_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_9
  simp only [← c79032, ← c79033, k2, ← c79034] at e_1285_9
  have hr := e_1285_9
  simp only [e_1285_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11893)) := by
  have c79035 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79036 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79037 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79038 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79039 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79040 := (publicBatchWrapper86_copies2470 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_10
  simp only [← c79035, ← c79036, ← c79037] at e_1285_10
  have e_1285_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_11
  simp only [← c79038, ← c79039, k2, ← c79040] at e_1285_11
  have hr := e_1285_11
  simp only [e_1285_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11894)) := by
  have c79041 := (publicBatchWrapper86_copies2470 a h).2.1
  have c79042 := (publicBatchWrapper86_copies2470 a h).2.2.1
  have c79043 := (publicBatchWrapper86_copies2470 a h).2.2.2.1
  have c79044 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.1
  have c79045 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.1
  have c79046 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_12
  simp only [← c79041, ← c79042, ← c79043] at e_1285_12
  have e_1285_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_13
  simp only [← c79044, ← c79045, k2, ← c79046] at e_1285_13
  have hr := e_1285_13
  simp only [e_1285_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11895)) := by
  have c79047 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.1
  have c79048 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.1
  have c79049 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.1
  have c79050 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79051 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79052 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_14
  simp only [← c79047, ← c79048, ← c79049] at e_1285_14
  have e_1285_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_15
  simp only [← c79050, ← c79051, k2, ← c79052] at e_1285_15
  have hr := e_1285_15
  simp only [e_1285_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11896)) := by
  have c79053 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79054 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79055 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79056 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79057 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79058 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_16
  simp only [← c79053, ← c79054, ← c79055] at e_1285_16
  have e_1285_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_17
  simp only [← c79056, ← c79057, k2, ← c79058] at e_1285_17
  have hr := e_1285_17
  simp only [e_1285_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11897)) := by
  have c79059 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79060 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79061 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79062 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79063 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79064 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_18
  simp only [← c79059, ← c79060, ← c79061] at e_1285_18
  have e_1285_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_19
  simp only [← c79062, ← c79063, k2, ← c79064] at e_1285_19
  have hr := e_1285_19
  simp only [e_1285_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11898)) := by
  have c79065 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79066 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79067 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79068 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79069 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79070 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_0
  simp only [← c79065, ← c79066, ← c79067] at e_1286_0
  have e_1286_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_1
  simp only [← c79068, ← c79069, k2, ← c79070] at e_1286_1
  have hr := e_1286_1
  simp only [e_1286_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11899)) := by
  have c79071 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79072 := (publicBatchWrapper86_copies2471 a h).1
  have c79073 := (publicBatchWrapper86_copies2471 a h).2.1
  have c79074 := (publicBatchWrapper86_copies2471 a h).2.2.1
  have c79075 := (publicBatchWrapper86_copies2471 a h).2.2.2.1
  have c79076 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_2
  simp only [← c79071, ← c79072, ← c79073] at e_1286_2
  have e_1286_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_3
  simp only [← c79074, ← c79075, k2, ← c79076] at e_1286_3
  have hr := e_1286_3
  simp only [e_1286_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11900)) := by
  have c79077 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.1
  have c79078 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.1
  have c79079 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.1
  have c79080 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.1
  have c79081 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.1
  have c79082 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_4
  simp only [← c79077, ← c79078, ← c79079] at e_1286_4
  have e_1286_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_5
  simp only [← c79080, ← c79081, k2, ← c79082] at e_1286_5
  have hr := e_1286_5
  simp only [e_1286_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11901)) := by
  have c79083 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79084 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79085 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79086 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79087 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79088 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_6
  simp only [← c79083, ← c79084, ← c79085] at e_1286_6
  have e_1286_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_7
  simp only [← c79086, ← c79087, k2, ← c79088] at e_1286_7
  have hr := e_1286_7
  simp only [e_1286_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11902)) := by
  have c79089 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79090 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79091 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79092 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79093 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79094 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_8
  simp only [← c79089, ← c79090, ← c79091] at e_1286_8
  have e_1286_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_9
  simp only [← c79092, ← c79093, k2, ← c79094] at e_1286_9
  have hr := e_1286_9
  simp only [e_1286_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11903)) := by
  have c79095 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79096 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79097 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79098 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79099 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79100 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_10
  simp only [← c79095, ← c79096, ← c79097] at e_1286_10
  have e_1286_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_11
  simp only [← c79098, ← c79099, k2, ← c79100] at e_1286_11
  have hr := e_1286_11
  simp only [e_1286_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11904)) := by
  have c79101 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79102 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79103 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79104 := (publicBatchWrapper86_copies2472 a h).1
  have c79105 := (publicBatchWrapper86_copies2472 a h).2.1
  have c79106 := (publicBatchWrapper86_copies2472 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_12
  simp only [← c79101, ← c79102, ← c79103] at e_1286_12
  have e_1286_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_13
  simp only [← c79104, ← c79105, k2, ← c79106] at e_1286_13
  have hr := e_1286_13
  simp only [e_1286_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11905)) := by
  have c79107 := (publicBatchWrapper86_copies2472 a h).2.2.2.1
  have c79108 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.1
  have c79109 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.1
  have c79110 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.1
  have c79111 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.1
  have c79112 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_14
  simp only [← c79107, ← c79108, ← c79109] at e_1286_14
  have e_1286_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_15
  simp only [← c79110, ← c79111, k2, ← c79112] at e_1286_15
  have hr := e_1286_15
  simp only [e_1286_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11906)) := by
  have c79113 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.1
  have c79114 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79115 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79116 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79117 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79118 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_16
  simp only [← c79113, ← c79114, ← c79115] at e_1286_16
  have e_1286_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_17
  simp only [← c79116, ← c79117, k2, ← c79118] at e_1286_17
  have hr := e_1286_17
  simp only [e_1286_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11907)) := by
  have c79119 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79120 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79121 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79122 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79123 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79124 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_18
  simp only [← c79119, ← c79120, ← c79121] at e_1286_18
  have e_1286_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_19
  simp only [← c79122, ← c79123, k2, ← c79124] at e_1286_19
  have hr := e_1286_19
  simp only [e_1286_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11908)) := by
  have c79125 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79126 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79127 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79128 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79129 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79130 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_0
  simp only [← c79125, ← c79126, ← c79127] at e_1287_0
  have e_1287_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_1
  simp only [← c79128, ← c79129, k2, ← c79130] at e_1287_1
  have hr := e_1287_1
  simp only [e_1287_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11909)) := by
  have c79131 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79132 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79133 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79134 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79135 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79136 := (publicBatchWrapper86_copies2473 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_2
  simp only [← c79131, ← c79132, ← c79133] at e_1287_2
  have e_1287_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_3
  simp only [← c79134, ← c79135, k2, ← c79136] at e_1287_3
  have hr := e_1287_3
  simp only [e_1287_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11910)) := by
  have c79137 := (publicBatchWrapper86_copies2473 a h).2.1
  have c79138 := (publicBatchWrapper86_copies2473 a h).2.2.1
  have c79139 := (publicBatchWrapper86_copies2473 a h).2.2.2.1
  have c79140 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.1
  have c79141 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.1
  have c79142 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_4
  simp only [← c79137, ← c79138, ← c79139] at e_1287_4
  have e_1287_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_5
  simp only [← c79140, ← c79141, k2, ← c79142] at e_1287_5
  have hr := e_1287_5
  simp only [e_1287_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11911)) := by
  have c79143 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.1
  have c79144 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.1
  have c79145 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.1
  have c79146 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79147 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79148 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_6
  simp only [← c79143, ← c79144, ← c79145] at e_1287_6
  have e_1287_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_7
  simp only [← c79146, ← c79147, k2, ← c79148] at e_1287_7
  have hr := e_1287_7
  simp only [e_1287_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11912)) := by
  have c79149 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79150 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79151 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79152 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79153 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79154 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_8
  simp only [← c79149, ← c79150, ← c79151] at e_1287_8
  have e_1287_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_9
  simp only [← c79152, ← c79153, k2, ← c79154] at e_1287_9
  have hr := e_1287_9
  simp only [e_1287_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11913)) := by
  have c79155 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79156 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79157 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79158 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79159 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79160 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_10
  simp only [← c79155, ← c79156, ← c79157] at e_1287_10
  have e_1287_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_11
  simp only [← c79158, ← c79159, k2, ← c79160] at e_1287_11
  have hr := e_1287_11
  simp only [e_1287_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11914)) := by
  have c79161 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79162 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79163 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79164 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79165 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79166 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_12
  simp only [← c79161, ← c79162, ← c79163] at e_1287_12
  have e_1287_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_13
  simp only [← c79164, ← c79165, k2, ← c79166] at e_1287_13
  have hr := e_1287_13
  simp only [e_1287_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11915)) := by
  have c79167 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79168 := (publicBatchWrapper86_copies2474 a h).1
  have c79169 := (publicBatchWrapper86_copies2474 a h).2.1
  have c79170 := (publicBatchWrapper86_copies2474 a h).2.2.1
  have c79171 := (publicBatchWrapper86_copies2474 a h).2.2.2.1
  have c79172 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_14
  simp only [← c79167, ← c79168, ← c79169] at e_1287_14
  have e_1287_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_15
  simp only [← c79170, ← c79171, k2, ← c79172] at e_1287_15
  have hr := e_1287_15
  simp only [e_1287_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11916)) := by
  have c79173 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.1
  have c79174 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.1
  have c79175 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.1
  have c79176 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.1
  have c79177 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.1
  have c79178 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_16
  simp only [← c79173, ← c79174, ← c79175] at e_1287_16
  have e_1287_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_17
  simp only [← c79176, ← c79177, k2, ← c79178] at e_1287_17
  have hr := e_1287_17
  simp only [e_1287_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11917)) := by
  have c79179 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79180 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79181 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79182 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79183 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79184 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_18
  simp only [← c79179, ← c79180, ← c79181] at e_1287_18
  have e_1287_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_19
  simp only [← c79182, ← c79183, k2, ← c79184] at e_1287_19
  have hr := e_1287_19
  simp only [e_1287_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11918)) := by
  have c79185 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79186 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79187 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79188 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79189 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79190 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_0
  simp only [← c79185, ← c79186, ← c79187] at e_1288_0
  have e_1288_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_1
  simp only [← c79188, ← c79189, k2, ← c79190] at e_1288_1
  have hr := e_1288_1
  simp only [e_1288_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11919)) := by
  have c79191 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79192 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79193 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79194 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79195 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79196 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_2
  simp only [← c79191, ← c79192, ← c79193] at e_1288_2
  have e_1288_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_3
  simp only [← c79194, ← c79195, k2, ← c79196] at e_1288_3
  have hr := e_1288_3
  simp only [e_1288_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11920)) := by
  have c79197 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79198 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79199 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79200 := (publicBatchWrapper86_copies2475 a h).1
  have c79201 := (publicBatchWrapper86_copies2475 a h).2.1
  have c79202 := (publicBatchWrapper86_copies2475 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_4
  simp only [← c79197, ← c79198, ← c79199] at e_1288_4
  have e_1288_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_5
  simp only [← c79200, ← c79201, k2, ← c79202] at e_1288_5
  have hr := e_1288_5
  simp only [e_1288_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11921)) := by
  have c79203 := (publicBatchWrapper86_copies2475 a h).2.2.2.1
  have c79204 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.1
  have c79205 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.1
  have c79206 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.1
  have c79207 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.1
  have c79208 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_6
  simp only [← c79203, ← c79204, ← c79205] at e_1288_6
  have e_1288_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_7
  simp only [← c79206, ← c79207, k2, ← c79208] at e_1288_7
  have hr := e_1288_7
  simp only [e_1288_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11922)) := by
  have c79209 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.1
  have c79210 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79211 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79212 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79213 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79214 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_8
  simp only [← c79209, ← c79210, ← c79211] at e_1288_8
  have e_1288_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_9
  simp only [← c79212, ← c79213, k2, ← c79214] at e_1288_9
  have hr := e_1288_9
  simp only [e_1288_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11923)) := by
  have c79215 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79216 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79217 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79218 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79219 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79220 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_10
  simp only [← c79215, ← c79216, ← c79217] at e_1288_10
  have e_1288_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_11
  simp only [← c79218, ← c79219, k2, ← c79220] at e_1288_11
  have hr := e_1288_11
  simp only [e_1288_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11924)) := by
  have c79221 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79222 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79223 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79224 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79225 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79226 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_12
  simp only [← c79221, ← c79222, ← c79223] at e_1288_12
  have e_1288_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_13
  simp only [← c79224, ← c79225, k2, ← c79226] at e_1288_13
  have hr := e_1288_13
  simp only [e_1288_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11925)) := by
  have c79227 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79228 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79229 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79230 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79231 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79232 := (publicBatchWrapper86_copies2476 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_14
  simp only [← c79227, ← c79228, ← c79229] at e_1288_14
  have e_1288_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_15
  simp only [← c79230, ← c79231, k2, ← c79232] at e_1288_15
  have hr := e_1288_15
  simp only [e_1288_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11926)) := by
  have c79233 := (publicBatchWrapper86_copies2476 a h).2.1
  have c79234 := (publicBatchWrapper86_copies2476 a h).2.2.1
  have c79235 := (publicBatchWrapper86_copies2476 a h).2.2.2.1
  have c79236 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.1
  have c79237 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.1
  have c79238 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_16
  simp only [← c79233, ← c79234, ← c79235] at e_1288_16
  have e_1288_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_17
  simp only [← c79236, ← c79237, k2, ← c79238] at e_1288_17
  have hr := e_1288_17
  simp only [e_1288_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11927)) := by
  have c79239 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.1
  have c79240 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.1
  have c79241 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.1
  have c79242 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79243 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79244 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_18
  simp only [← c79239, ← c79240, ← c79241] at e_1288_18
  have e_1288_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_19
  simp only [← c79242, ← c79243, k2, ← c79244] at e_1288_19
  have hr := e_1288_19
  simp only [e_1288_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11928)) := by
  have c79245 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79246 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79247 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79248 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79249 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79250 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_0
  simp only [← c79245, ← c79246, ← c79247] at e_1289_0
  have e_1289_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_1
  simp only [← c79248, ← c79249, k2, ← c79250] at e_1289_1
  have hr := e_1289_1
  simp only [e_1289_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11929)) := by
  have c79251 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79252 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79253 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79254 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79255 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79256 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_2
  simp only [← c79251, ← c79252, ← c79253] at e_1289_2
  have e_1289_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_3
  simp only [← c79254, ← c79255, k2, ← c79256] at e_1289_3
  have hr := e_1289_3
  simp only [e_1289_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11930)) := by
  have c79257 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79258 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79259 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79260 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79261 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79262 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_4
  simp only [← c79257, ← c79258, ← c79259] at e_1289_4
  have e_1289_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_5
  simp only [← c79260, ← c79261, k2, ← c79262] at e_1289_5
  have hr := e_1289_5
  simp only [e_1289_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11931)) := by
  have c79263 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79264 := (publicBatchWrapper86_copies2477 a h).1
  have c79265 := (publicBatchWrapper86_copies2477 a h).2.1
  have c79266 := (publicBatchWrapper86_copies2477 a h).2.2.1
  have c79267 := (publicBatchWrapper86_copies2477 a h).2.2.2.1
  have c79268 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_6
  simp only [← c79263, ← c79264, ← c79265] at e_1289_6
  have e_1289_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_7
  simp only [← c79266, ← c79267, k2, ← c79268] at e_1289_7
  have hr := e_1289_7
  simp only [e_1289_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11932)) := by
  have c79269 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.1
  have c79270 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.1
  have c79271 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.1
  have c79272 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.1
  have c79273 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.1
  have c79274 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_8
  simp only [← c79269, ← c79270, ← c79271] at e_1289_8
  have e_1289_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_9
  simp only [← c79272, ← c79273, k2, ← c79274] at e_1289_9
  have hr := e_1289_9
  simp only [e_1289_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11933)) := by
  have c79275 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79276 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79277 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79278 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79279 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79280 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_10
  simp only [← c79275, ← c79276, ← c79277] at e_1289_10
  have e_1289_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_11
  simp only [← c79278, ← c79279, k2, ← c79280] at e_1289_11
  have hr := e_1289_11
  simp only [e_1289_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11934)) := by
  have c79281 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79282 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79283 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79284 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79285 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79286 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_12
  simp only [← c79281, ← c79282, ← c79283] at e_1289_12
  have e_1289_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_13
  simp only [← c79284, ← c79285, k2, ← c79286] at e_1289_13
  have hr := e_1289_13
  simp only [e_1289_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11935)) := by
  have c79287 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79288 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79289 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79290 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79291 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79292 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_14
  simp only [← c79287, ← c79288, ← c79289] at e_1289_14
  have e_1289_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_15
  simp only [← c79290, ← c79291, k2, ← c79292] at e_1289_15
  have hr := e_1289_15
  simp only [e_1289_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11936)) := by
  have c79293 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79294 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79295 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79296 := (publicBatchWrapper86_copies2478 a h).1
  have c79297 := (publicBatchWrapper86_copies2478 a h).2.1
  have c79298 := (publicBatchWrapper86_copies2478 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_16
  simp only [← c79293, ← c79294, ← c79295] at e_1289_16
  have e_1289_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_17
  simp only [← c79296, ← c79297, k2, ← c79298] at e_1289_17
  have hr := e_1289_17
  simp only [e_1289_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11937)) := by
  have c79299 := (publicBatchWrapper86_copies2478 a h).2.2.2.1
  have c79300 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.1
  have c79301 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.1
  have c79302 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.1
  have c79303 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.1
  have c79304 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_18
  simp only [← c79299, ← c79300, ← c79301] at e_1289_18
  have e_1289_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_19
  simp only [← c79302, ← c79303, k2, ← c79304] at e_1289_19
  have hr := e_1289_19
  simp only [e_1289_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11938)) := by
  have c79305 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.1
  have c79306 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79307 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79308 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79309 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79310 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_0
  simp only [← c79305, ← c79306, ← c79307] at e_1290_0
  have e_1290_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_1
  simp only [← c79308, ← c79309, k2, ← c79310] at e_1290_1
  have hr := e_1290_1
  simp only [e_1290_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11939)) := by
  have c79311 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79312 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79313 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79314 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79315 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79316 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_2
  simp only [← c79311, ← c79312, ← c79313] at e_1290_2
  have e_1290_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_3
  simp only [← c79314, ← c79315, k2, ← c79316] at e_1290_3
  have hr := e_1290_3
  simp only [e_1290_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11940)) := by
  have c79317 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79318 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79319 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79320 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79321 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79322 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_4
  simp only [← c79317, ← c79318, ← c79319] at e_1290_4
  have e_1290_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_5
  simp only [← c79320, ← c79321, k2, ← c79322] at e_1290_5
  have hr := e_1290_5
  simp only [e_1290_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11941)) := by
  have c79323 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79324 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79325 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79326 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79327 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79328 := (publicBatchWrapper86_copies2479 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_6
  simp only [← c79323, ← c79324, ← c79325] at e_1290_6
  have e_1290_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_7
  simp only [← c79326, ← c79327, k2, ← c79328] at e_1290_7
  have hr := e_1290_7
  simp only [e_1290_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11942)) := by
  have c79329 := (publicBatchWrapper86_copies2479 a h).2.1
  have c79330 := (publicBatchWrapper86_copies2479 a h).2.2.1
  have c79331 := (publicBatchWrapper86_copies2479 a h).2.2.2.1
  have c79332 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.1
  have c79333 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.1
  have c79334 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_8
  simp only [← c79329, ← c79330, ← c79331] at e_1290_8
  have e_1290_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_9
  simp only [← c79332, ← c79333, k2, ← c79334] at e_1290_9
  have hr := e_1290_9
  simp only [e_1290_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11943)) := by
  have c79335 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.1
  have c79336 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.1
  have c79337 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.1
  have c79338 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79339 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79340 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_10
  simp only [← c79335, ← c79336, ← c79337] at e_1290_10
  have e_1290_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_11
  simp only [← c79338, ← c79339, k2, ← c79340] at e_1290_11
  have hr := e_1290_11
  simp only [e_1290_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11944)) := by
  have c79341 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79342 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79343 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79344 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79345 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79346 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_12
  simp only [← c79341, ← c79342, ← c79343] at e_1290_12
  have e_1290_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_13
  simp only [← c79344, ← c79345, k2, ← c79346] at e_1290_13
  have hr := e_1290_13
  simp only [e_1290_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11945)) := by
  have c79347 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79348 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79349 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79350 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79351 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79352 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_14
  simp only [← c79347, ← c79348, ← c79349] at e_1290_14
  have e_1290_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_15
  simp only [← c79350, ← c79351, k2, ← c79352] at e_1290_15
  have hr := e_1290_15
  simp only [e_1290_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11946)) := by
  have c79353 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79354 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79355 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79356 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79357 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79358 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_16
  simp only [← c79353, ← c79354, ← c79355] at e_1290_16
  have e_1290_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_17
  simp only [← c79356, ← c79357, k2, ← c79358] at e_1290_17
  have hr := e_1290_17
  simp only [e_1290_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11947)) := by
  have c79359 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79360 := (publicBatchWrapper86_copies2480 a h).1
  have c79361 := (publicBatchWrapper86_copies2480 a h).2.1
  have c79362 := (publicBatchWrapper86_copies2480 a h).2.2.1
  have c79363 := (publicBatchWrapper86_copies2480 a h).2.2.2.1
  have c79364 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_18
  simp only [← c79359, ← c79360, ← c79361] at e_1290_18
  have e_1290_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_19
  simp only [← c79362, ← c79363, k2, ← c79364] at e_1290_19
  have hr := e_1290_19
  simp only [e_1290_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11948)) := by
  have c79365 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.1
  have c79366 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.1
  have c79367 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.1
  have c79368 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.1
  have c79369 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.1
  have c79370 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_0
  simp only [← c79365, ← c79366, ← c79367] at e_1291_0
  have e_1291_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_1
  simp only [← c79368, ← c79369, k2, ← c79370] at e_1291_1
  have hr := e_1291_1
  simp only [e_1291_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11949)) := by
  have c79371 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79372 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79373 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79374 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79375 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79376 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_2
  simp only [← c79371, ← c79372, ← c79373] at e_1291_2
  have e_1291_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_3
  simp only [← c79374, ← c79375, k2, ← c79376] at e_1291_3
  have hr := e_1291_3
  simp only [e_1291_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11950)) := by
  have c79377 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79378 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79379 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79380 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79381 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79382 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_4
  simp only [← c79377, ← c79378, ← c79379] at e_1291_4
  have e_1291_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_5
  simp only [← c79380, ← c79381, k2, ← c79382] at e_1291_5
  have hr := e_1291_5
  simp only [e_1291_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11951)) := by
  have c79383 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79384 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79385 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79386 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79387 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79388 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_6
  simp only [← c79383, ← c79384, ← c79385] at e_1291_6
  have e_1291_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_7
  simp only [← c79386, ← c79387, k2, ← c79388] at e_1291_7
  have hr := e_1291_7
  simp only [e_1291_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11952)) := by
  have c79389 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79390 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79391 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79392 := (publicBatchWrapper86_copies2481 a h).1
  have c79393 := (publicBatchWrapper86_copies2481 a h).2.1
  have c79394 := (publicBatchWrapper86_copies2481 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_8
  simp only [← c79389, ← c79390, ← c79391] at e_1291_8
  have e_1291_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_9
  simp only [← c79392, ← c79393, k2, ← c79394] at e_1291_9
  have hr := e_1291_9
  simp only [e_1291_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11953)) := by
  have c79395 := (publicBatchWrapper86_copies2481 a h).2.2.2.1
  have c79396 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.1
  have c79397 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.1
  have c79398 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.1
  have c79399 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.1
  have c79400 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_10
  simp only [← c79395, ← c79396, ← c79397] at e_1291_10
  have e_1291_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_11
  simp only [← c79398, ← c79399, k2, ← c79400] at e_1291_11
  have hr := e_1291_11
  simp only [e_1291_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11954)) := by
  have c79401 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.1
  have c79402 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79403 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79404 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79405 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79406 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_12
  simp only [← c79401, ← c79402, ← c79403] at e_1291_12
  have e_1291_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_13
  simp only [← c79404, ← c79405, k2, ← c79406] at e_1291_13
  have hr := e_1291_13
  simp only [e_1291_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11955)) := by
  have c79407 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79408 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79409 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79410 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79411 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79412 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_14
  simp only [← c79407, ← c79408, ← c79409] at e_1291_14
  have e_1291_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_15
  simp only [← c79410, ← c79411, k2, ← c79412] at e_1291_15
  have hr := e_1291_15
  simp only [e_1291_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11956)) := by
  have c79413 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79414 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79415 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79416 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79417 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79418 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_16
  simp only [← c79413, ← c79414, ← c79415] at e_1291_16
  have e_1291_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_17
  simp only [← c79416, ← c79417, k2, ← c79418] at e_1291_17
  have hr := e_1291_17
  simp only [e_1291_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11957)) := by
  have c79419 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79420 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79421 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79422 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79423 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79424 := (publicBatchWrapper86_copies2482 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_18
  simp only [← c79419, ← c79420, ← c79421] at e_1291_18
  have e_1291_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_19
  simp only [← c79422, ← c79423, k2, ← c79424] at e_1291_19
  have hr := e_1291_19
  simp only [e_1291_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11958)) := by
  have c79425 := (publicBatchWrapper86_copies2482 a h).2.1
  have c79426 := (publicBatchWrapper86_copies2482 a h).2.2.1
  have c79427 := (publicBatchWrapper86_copies2482 a h).2.2.2.1
  have c79428 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.1
  have c79429 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.1
  have c79430 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_0
  simp only [← c79425, ← c79426, ← c79427] at e_1292_0
  have e_1292_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_1
  simp only [← c79428, ← c79429, k2, ← c79430] at e_1292_1
  have hr := e_1292_1
  simp only [e_1292_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11959)) := by
  have c79431 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.1
  have c79432 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.1
  have c79433 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.1
  have c79434 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79435 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79436 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_2
  simp only [← c79431, ← c79432, ← c79433] at e_1292_2
  have e_1292_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_3
  simp only [← c79434, ← c79435, k2, ← c79436] at e_1292_3
  have hr := e_1292_3
  simp only [e_1292_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11960)) := by
  have c79437 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79438 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79439 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79440 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79441 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79442 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_4
  simp only [← c79437, ← c79438, ← c79439] at e_1292_4
  have e_1292_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_5
  simp only [← c79440, ← c79441, k2, ← c79442] at e_1292_5
  have hr := e_1292_5
  simp only [e_1292_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11961)) := by
  have c79443 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79444 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79445 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79446 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79447 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79448 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_6
  simp only [← c79443, ← c79444, ← c79445] at e_1292_6
  have e_1292_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_7
  simp only [← c79446, ← c79447, k2, ← c79448] at e_1292_7
  have hr := e_1292_7
  simp only [e_1292_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11962)) := by
  have c79449 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79450 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79451 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79452 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79453 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79454 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_8
  simp only [← c79449, ← c79450, ← c79451] at e_1292_8
  have e_1292_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_9
  simp only [← c79452, ← c79453, k2, ← c79454] at e_1292_9
  have hr := e_1292_9
  simp only [e_1292_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11963)) := by
  have c79455 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79456 := (publicBatchWrapper86_copies2483 a h).1
  have c79457 := (publicBatchWrapper86_copies2483 a h).2.1
  have c79458 := (publicBatchWrapper86_copies2483 a h).2.2.1
  have c79459 := (publicBatchWrapper86_copies2483 a h).2.2.2.1
  have c79460 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_10
  simp only [← c79455, ← c79456, ← c79457] at e_1292_10
  have e_1292_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_11
  simp only [← c79458, ← c79459, k2, ← c79460] at e_1292_11
  have hr := e_1292_11
  simp only [e_1292_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11964)) := by
  have c79461 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.1
  have c79462 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.1
  have c79463 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.1
  have c79464 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.1
  have c79465 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.1
  have c79466 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_12
  simp only [← c79461, ← c79462, ← c79463] at e_1292_12
  have e_1292_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_13
  simp only [← c79464, ← c79465, k2, ← c79466] at e_1292_13
  have hr := e_1292_13
  simp only [e_1292_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11965)) := by
  have c79467 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79468 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79469 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79470 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79471 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79472 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_14
  simp only [← c79467, ← c79468, ← c79469] at e_1292_14
  have e_1292_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_15
  simp only [← c79470, ← c79471, k2, ← c79472] at e_1292_15
  have hr := e_1292_15
  simp only [e_1292_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11966)) := by
  have c79473 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79474 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79475 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79476 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79477 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79478 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_16
  simp only [← c79473, ← c79474, ← c79475] at e_1292_16
  have e_1292_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_17
  simp only [← c79476, ← c79477, k2, ← c79478] at e_1292_17
  have hr := e_1292_17
  simp only [e_1292_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11967)) := by
  have c79479 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79480 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79481 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79482 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79483 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79484 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_18
  simp only [← c79479, ← c79480, ← c79481] at e_1292_18
  have e_1292_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_19
  simp only [← c79482, ← c79483, k2, ← c79484] at e_1292_19
  have hr := e_1292_19
  simp only [e_1292_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11968)) := by
  have c79485 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79486 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79487 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79488 := (publicBatchWrapper86_copies2484 a h).1
  have c79489 := (publicBatchWrapper86_copies2484 a h).2.1
  have c79490 := (publicBatchWrapper86_copies2484 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_0
  simp only [← c79485, ← c79486, ← c79487] at e_1293_0
  have e_1293_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_1
  simp only [← c79488, ← c79489, k2, ← c79490] at e_1293_1
  have hr := e_1293_1
  simp only [e_1293_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11969)) := by
  have c79491 := (publicBatchWrapper86_copies2484 a h).2.2.2.1
  have c79492 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.1
  have c79493 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.1
  have c79494 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.1
  have c79495 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.1
  have c79496 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_2
  simp only [← c79491, ← c79492, ← c79493] at e_1293_2
  have e_1293_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_3
  simp only [← c79494, ← c79495, k2, ← c79496] at e_1293_3
  have hr := e_1293_3
  simp only [e_1293_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12010)) := by
  have c79497 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.1
  have c79498 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79499 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79500 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79501 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79502 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_4
  simp only [← c79497, ← c79498, ← c79499] at e_1293_4
  have e_1293_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_5
  simp only [← c79500, ← c79501, k2, ← c79502] at e_1293_5
  have hr := e_1293_5
  simp only [e_1293_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12011)) := by
  have c79503 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79504 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79505 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79506 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79507 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79508 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_6
  simp only [← c79503, ← c79504, ← c79505] at e_1293_6
  have e_1293_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_7
  simp only [← c79506, ← c79507, k2, ← c79508] at e_1293_7
  have hr := e_1293_7
  simp only [e_1293_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12012)) := by
  have c79509 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79510 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79511 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79512 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79513 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79514 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_8
  simp only [← c79509, ← c79510, ← c79511] at e_1293_8
  have e_1293_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_9
  simp only [← c79512, ← c79513, k2, ← c79514] at e_1293_9
  have hr := e_1293_9
  simp only [e_1293_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12013)) := by
  have c79515 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79516 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79517 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79518 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79519 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79520 := (publicBatchWrapper86_copies2485 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_10
  simp only [← c79515, ← c79516, ← c79517] at e_1293_10
  have e_1293_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_11
  simp only [← c79518, ← c79519, k2, ← c79520] at e_1293_11
  have hr := e_1293_11
  simp only [e_1293_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12014)) := by
  have c79521 := (publicBatchWrapper86_copies2485 a h).2.1
  have c79522 := (publicBatchWrapper86_copies2485 a h).2.2.1
  have c79523 := (publicBatchWrapper86_copies2485 a h).2.2.2.1
  have c79524 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.1
  have c79525 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.1
  have c79526 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_12
  simp only [← c79521, ← c79522, ← c79523] at e_1293_12
  have e_1293_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_13
  simp only [← c79524, ← c79525, k2, ← c79526] at e_1293_13
  have hr := e_1293_13
  simp only [e_1293_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12015)) := by
  have c79527 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.1
  have c79528 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.1
  have c79529 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.1
  have c79530 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79531 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79532 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_14
  simp only [← c79527, ← c79528, ← c79529] at e_1293_14
  have e_1293_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_15
  simp only [← c79530, ← c79531, k2, ← c79532] at e_1293_15
  have hr := e_1293_15
  simp only [e_1293_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12016)) := by
  have c79533 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79534 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79535 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79536 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79537 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79538 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_16
  simp only [← c79533, ← c79534, ← c79535] at e_1293_16
  have e_1293_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_17
  simp only [← c79536, ← c79537, k2, ← c79538] at e_1293_17
  have hr := e_1293_17
  simp only [e_1293_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12017)) := by
  have c79539 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79540 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79541 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79542 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79543 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79544 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_18
  simp only [← c79539, ← c79540, ← c79541] at e_1293_18
  have e_1293_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_19
  simp only [← c79542, ← c79543, k2, ← c79544] at e_1293_19
  have hr := e_1293_19
  simp only [e_1293_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12018)) := by
  have c79545 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79546 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79547 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79548 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79549 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79550 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_0
  simp only [← c79545, ← c79546, ← c79547] at e_1294_0
  have e_1294_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_1
  simp only [← c79548, ← c79549, k2, ← c79550] at e_1294_1
  have hr := e_1294_1
  simp only [e_1294_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12019)) := by
  have c79551 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79552 := (publicBatchWrapper86_copies2486 a h).1
  have c79553 := (publicBatchWrapper86_copies2486 a h).2.1
  have c79554 := (publicBatchWrapper86_copies2486 a h).2.2.1
  have c79555 := (publicBatchWrapper86_copies2486 a h).2.2.2.1
  have c79556 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_2
  simp only [← c79551, ← c79552, ← c79553] at e_1294_2
  have e_1294_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_3
  simp only [← c79554, ← c79555, k2, ← c79556] at e_1294_3
  have hr := e_1294_3
  simp only [e_1294_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12020)) := by
  have c79557 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.1
  have c79558 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.1
  have c79559 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.1
  have c79560 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.1
  have c79561 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.1
  have c79562 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_4
  simp only [← c79557, ← c79558, ← c79559] at e_1294_4
  have e_1294_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_5
  simp only [← c79560, ← c79561, k2, ← c79562] at e_1294_5
  have hr := e_1294_5
  simp only [e_1294_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12021)) := by
  have c79563 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79564 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79565 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79566 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79567 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79568 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_6
  simp only [← c79563, ← c79564, ← c79565] at e_1294_6
  have e_1294_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_7
  simp only [← c79566, ← c79567, k2, ← c79568] at e_1294_7
  have hr := e_1294_7
  simp only [e_1294_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12022)) := by
  have c79569 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79570 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79571 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79572 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79573 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79574 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_8
  simp only [← c79569, ← c79570, ← c79571] at e_1294_8
  have e_1294_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_9
  simp only [← c79572, ← c79573, k2, ← c79574] at e_1294_9
  have hr := e_1294_9
  simp only [e_1294_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12023)) := by
  have c79575 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79576 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79577 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79578 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79579 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79580 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_10
  simp only [← c79575, ← c79576, ← c79577] at e_1294_10
  have e_1294_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_11
  simp only [← c79578, ← c79579, k2, ← c79580] at e_1294_11
  have hr := e_1294_11
  simp only [e_1294_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12024)) := by
  have c79581 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79582 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79583 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79584 := (publicBatchWrapper86_copies2487 a h).1
  have c79585 := (publicBatchWrapper86_copies2487 a h).2.1
  have c79586 := (publicBatchWrapper86_copies2487 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_12
  simp only [← c79581, ← c79582, ← c79583] at e_1294_12
  have e_1294_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_13
  simp only [← c79584, ← c79585, k2, ← c79586] at e_1294_13
  have hr := e_1294_13
  simp only [e_1294_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12025)) := by
  have c79587 := (publicBatchWrapper86_copies2487 a h).2.2.2.1
  have c79588 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.1
  have c79589 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.1
  have c79590 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.1
  have c79591 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.1
  have c79592 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_14
  simp only [← c79587, ← c79588, ← c79589] at e_1294_14
  have e_1294_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_15
  simp only [← c79590, ← c79591, k2, ← c79592] at e_1294_15
  have hr := e_1294_15
  simp only [e_1294_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12026)) := by
  have c79593 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.1
  have c79594 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79595 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79596 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79597 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79598 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_16
  simp only [← c79593, ← c79594, ← c79595] at e_1294_16
  have e_1294_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_17
  simp only [← c79596, ← c79597, k2, ← c79598] at e_1294_17
  have hr := e_1294_17
  simp only [e_1294_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12027)) := by
  have c79599 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79600 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79601 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79602 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79603 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79604 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_18
  simp only [← c79599, ← c79600, ← c79601] at e_1294_18
  have e_1294_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_19
  simp only [← c79602, ← c79603, k2, ← c79604] at e_1294_19
  have hr := e_1294_19
  simp only [e_1294_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12028)) := by
  have c79605 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79606 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79607 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79608 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79609 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79610 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_0
  simp only [← c79605, ← c79606, ← c79607] at e_1295_0
  have e_1295_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_1
  simp only [← c79608, ← c79609, k2, ← c79610] at e_1295_1
  have hr := e_1295_1
  simp only [e_1295_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12029)) := by
  have c79611 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79612 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79613 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79614 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79615 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79616 := (publicBatchWrapper86_copies2488 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_2
  simp only [← c79611, ← c79612, ← c79613] at e_1295_2
  have e_1295_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_3
  simp only [← c79614, ← c79615, k2, ← c79616] at e_1295_3
  have hr := e_1295_3
  simp only [e_1295_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12030)) := by
  have c79617 := (publicBatchWrapper86_copies2488 a h).2.1
  have c79618 := (publicBatchWrapper86_copies2488 a h).2.2.1
  have c79619 := (publicBatchWrapper86_copies2488 a h).2.2.2.1
  have c79620 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.1
  have c79621 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.1
  have c79622 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_4
  simp only [← c79617, ← c79618, ← c79619] at e_1295_4
  have e_1295_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_5
  simp only [← c79620, ← c79621, k2, ← c79622] at e_1295_5
  have hr := e_1295_5
  simp only [e_1295_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12031)) := by
  have c79623 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.1
  have c79624 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.1
  have c79625 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.1
  have c79626 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79627 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79628 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_6
  simp only [← c79623, ← c79624, ← c79625] at e_1295_6
  have e_1295_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_7
  simp only [← c79626, ← c79627, k2, ← c79628] at e_1295_7
  have hr := e_1295_7
  simp only [e_1295_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12032)) := by
  have c79629 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79630 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79631 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79632 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79633 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79634 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_8
  simp only [← c79629, ← c79630, ← c79631] at e_1295_8
  have e_1295_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_9
  simp only [← c79632, ← c79633, k2, ← c79634] at e_1295_9
  have hr := e_1295_9
  simp only [e_1295_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12033)) := by
  have c79635 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79636 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79637 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79638 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79639 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79640 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_10
  simp only [← c79635, ← c79636, ← c79637] at e_1295_10
  have e_1295_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_11
  simp only [← c79638, ← c79639, k2, ← c79640] at e_1295_11
  have hr := e_1295_11
  simp only [e_1295_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12034)) := by
  have c79641 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79642 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79643 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79644 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79645 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79646 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_12
  simp only [← c79641, ← c79642, ← c79643] at e_1295_12
  have e_1295_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_13
  simp only [← c79644, ← c79645, k2, ← c79646] at e_1295_13
  have hr := e_1295_13
  simp only [e_1295_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12035)) := by
  have c79647 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79648 := (publicBatchWrapper86_copies2489 a h).1
  have c79649 := (publicBatchWrapper86_copies2489 a h).2.1
  have c79650 := (publicBatchWrapper86_copies2489 a h).2.2.1
  have c79651 := (publicBatchWrapper86_copies2489 a h).2.2.2.1
  have c79652 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_14
  simp only [← c79647, ← c79648, ← c79649] at e_1295_14
  have e_1295_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_15
  simp only [← c79650, ← c79651, k2, ← c79652] at e_1295_15
  have hr := e_1295_15
  simp only [e_1295_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12036)) := by
  have c79653 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.1
  have c79654 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.1
  have c79655 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.1
  have c79656 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.1
  have c79657 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.1
  have c79658 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_16
  simp only [← c79653, ← c79654, ← c79655] at e_1295_16
  have e_1295_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_17
  simp only [← c79656, ← c79657, k2, ← c79658] at e_1295_17
  have hr := e_1295_17
  simp only [e_1295_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12037)) := by
  have c79659 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79660 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79661 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79662 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79663 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79664 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_18
  simp only [← c79659, ← c79660, ← c79661] at e_1295_18
  have e_1295_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_19
  simp only [← c79662, ← c79663, k2, ← c79664] at e_1295_19
  have hr := e_1295_19
  simp only [e_1295_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12038)) := by
  have c79665 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79666 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79667 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79668 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79669 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79670 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_0
  simp only [← c79665, ← c79666, ← c79667] at e_1296_0
  have e_1296_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_1
  simp only [← c79668, ← c79669, k2, ← c79670] at e_1296_1
  have hr := e_1296_1
  simp only [e_1296_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12039)) := by
  have c79671 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79672 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79673 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79674 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79675 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79676 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_2
  simp only [← c79671, ← c79672, ← c79673] at e_1296_2
  have e_1296_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_3
  simp only [← c79674, ← c79675, k2, ← c79676] at e_1296_3
  have hr := e_1296_3
  simp only [e_1296_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12040)) := by
  have c79677 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79678 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79679 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79680 := (publicBatchWrapper86_copies2490 a h).1
  have c79681 := (publicBatchWrapper86_copies2490 a h).2.1
  have c79682 := (publicBatchWrapper86_copies2490 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_4
  simp only [← c79677, ← c79678, ← c79679] at e_1296_4
  have e_1296_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_5
  simp only [← c79680, ← c79681, k2, ← c79682] at e_1296_5
  have hr := e_1296_5
  simp only [e_1296_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12041)) := by
  have c79683 := (publicBatchWrapper86_copies2490 a h).2.2.2.1
  have c79684 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.1
  have c79685 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.1
  have c79686 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.1
  have c79687 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.1
  have c79688 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_6
  simp only [← c79683, ← c79684, ← c79685] at e_1296_6
  have e_1296_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_7
  simp only [← c79686, ← c79687, k2, ← c79688] at e_1296_7
  have hr := e_1296_7
  simp only [e_1296_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12042)) := by
  have c79689 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.1
  have c79690 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79691 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79692 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79693 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79694 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_8
  simp only [← c79689, ← c79690, ← c79691] at e_1296_8
  have e_1296_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_9
  simp only [← c79692, ← c79693, k2, ← c79694] at e_1296_9
  have hr := e_1296_9
  simp only [e_1296_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12043)) := by
  have c79695 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79696 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79697 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79698 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79699 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79700 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_10
  simp only [← c79695, ← c79696, ← c79697] at e_1296_10
  have e_1296_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_11
  simp only [← c79698, ← c79699, k2, ← c79700] at e_1296_11
  have hr := e_1296_11
  simp only [e_1296_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12044)) := by
  have c79701 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79702 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79703 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79704 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79705 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79706 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_12
  simp only [← c79701, ← c79702, ← c79703] at e_1296_12
  have e_1296_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_13
  simp only [← c79704, ← c79705, k2, ← c79706] at e_1296_13
  have hr := e_1296_13
  simp only [e_1296_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12045)) := by
  have c79707 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79708 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79709 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79710 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79711 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79712 := (publicBatchWrapper86_copies2491 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_14
  simp only [← c79707, ← c79708, ← c79709] at e_1296_14
  have e_1296_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_15
  simp only [← c79710, ← c79711, k2, ← c79712] at e_1296_15
  have hr := e_1296_15
  simp only [e_1296_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12046)) := by
  have c79713 := (publicBatchWrapper86_copies2491 a h).2.1
  have c79714 := (publicBatchWrapper86_copies2491 a h).2.2.1
  have c79715 := (publicBatchWrapper86_copies2491 a h).2.2.2.1
  have c79716 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.1
  have c79717 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.1
  have c79718 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_16
  simp only [← c79713, ← c79714, ← c79715] at e_1296_16
  have e_1296_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_17
  simp only [← c79716, ← c79717, k2, ← c79718] at e_1296_17
  have hr := e_1296_17
  simp only [e_1296_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12047)) := by
  have c79719 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.1
  have c79720 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.1
  have c79721 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.1
  have c79722 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79723 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79724 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_18
  simp only [← c79719, ← c79720, ← c79721] at e_1296_18
  have e_1296_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_19
  simp only [← c79722, ← c79723, k2, ← c79724] at e_1296_19
  have hr := e_1296_19
  simp only [e_1296_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12048)) := by
  have c79725 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79726 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79727 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79728 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79729 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79730 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_0
  simp only [← c79725, ← c79726, ← c79727] at e_1297_0
  have e_1297_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_1
  simp only [← c79728, ← c79729, k2, ← c79730] at e_1297_1
  have hr := e_1297_1
  simp only [e_1297_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12049)) := by
  have c79731 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79732 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79733 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79734 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79735 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79736 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_2
  simp only [← c79731, ← c79732, ← c79733] at e_1297_2
  have e_1297_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_3
  simp only [← c79734, ← c79735, k2, ← c79736] at e_1297_3
  have hr := e_1297_3
  simp only [e_1297_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12050)) := by
  have c79737 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79738 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79739 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79740 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79741 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79742 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_4
  simp only [← c79737, ← c79738, ← c79739] at e_1297_4
  have e_1297_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_5
  simp only [← c79740, ← c79741, k2, ← c79742] at e_1297_5
  have hr := e_1297_5
  simp only [e_1297_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12051)) := by
  have c79743 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79744 := (publicBatchWrapper86_copies2492 a h).1
  have c79745 := (publicBatchWrapper86_copies2492 a h).2.1
  have c79746 := (publicBatchWrapper86_copies2492 a h).2.2.1
  have c79747 := (publicBatchWrapper86_copies2492 a h).2.2.2.1
  have c79748 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_6
  simp only [← c79743, ← c79744, ← c79745] at e_1297_6
  have e_1297_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_7
  simp only [← c79746, ← c79747, k2, ← c79748] at e_1297_7
  have hr := e_1297_7
  simp only [e_1297_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12052)) := by
  have c79749 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.1
  have c79750 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.1
  have c79751 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.1
  have c79752 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.1
  have c79753 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.1
  have c79754 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_8
  simp only [← c79749, ← c79750, ← c79751] at e_1297_8
  have e_1297_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_9
  simp only [← c79752, ← c79753, k2, ← c79754] at e_1297_9
  have hr := e_1297_9
  simp only [e_1297_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12053)) := by
  have c79755 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79756 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79757 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79758 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79759 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79760 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_10
  simp only [← c79755, ← c79756, ← c79757] at e_1297_10
  have e_1297_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_11
  simp only [← c79758, ← c79759, k2, ← c79760] at e_1297_11
  have hr := e_1297_11
  simp only [e_1297_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12054)) := by
  have c79761 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79762 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79763 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79764 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79765 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79766 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_12
  simp only [← c79761, ← c79762, ← c79763] at e_1297_12
  have e_1297_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_13
  simp only [← c79764, ← c79765, k2, ← c79766] at e_1297_13
  have hr := e_1297_13
  simp only [e_1297_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12055)) := by
  have c79767 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79768 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79769 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79770 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79771 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79772 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_14
  simp only [← c79767, ← c79768, ← c79769] at e_1297_14
  have e_1297_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_15
  simp only [← c79770, ← c79771, k2, ← c79772] at e_1297_15
  have hr := e_1297_15
  simp only [e_1297_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12056)) := by
  have c79773 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79774 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79775 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79776 := (publicBatchWrapper86_copies2493 a h).1
  have c79777 := (publicBatchWrapper86_copies2493 a h).2.1
  have c79778 := (publicBatchWrapper86_copies2493 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_16
  simp only [← c79773, ← c79774, ← c79775] at e_1297_16
  have e_1297_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_17
  simp only [← c79776, ← c79777, k2, ← c79778] at e_1297_17
  have hr := e_1297_17
  simp only [e_1297_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12057)) := by
  have c79779 := (publicBatchWrapper86_copies2493 a h).2.2.2.1
  have c79780 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.1
  have c79781 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.1
  have c79782 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.1
  have c79783 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.1
  have c79784 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_18
  simp only [← c79779, ← c79780, ← c79781] at e_1297_18
  have e_1297_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_19
  simp only [← c79782, ← c79783, k2, ← c79784] at e_1297_19
  have hr := e_1297_19
  simp only [e_1297_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12058)) := by
  have c79785 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.1
  have c79786 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79787 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79788 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79789 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79790 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_0
  simp only [← c79785, ← c79786, ← c79787] at e_1298_0
  have e_1298_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_1
  simp only [← c79788, ← c79789, k2, ← c79790] at e_1298_1
  have hr := e_1298_1
  simp only [e_1298_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12059)) := by
  have c79791 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79792 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79793 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79794 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79795 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79796 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_2
  simp only [← c79791, ← c79792, ← c79793] at e_1298_2
  have e_1298_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_3
  simp only [← c79794, ← c79795, k2, ← c79796] at e_1298_3
  have hr := e_1298_3
  simp only [e_1298_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12060)) := by
  have c79797 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79798 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79799 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79800 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79801 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79802 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_4
  simp only [← c79797, ← c79798, ← c79799] at e_1298_4
  have e_1298_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_5
  simp only [← c79800, ← c79801, k2, ← c79802] at e_1298_5
  have hr := e_1298_5
  simp only [e_1298_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12061)) := by
  have c79803 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79804 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79805 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79806 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79807 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79808 := (publicBatchWrapper86_copies2494 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_6
  simp only [← c79803, ← c79804, ← c79805] at e_1298_6
  have e_1298_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_7
  simp only [← c79806, ← c79807, k2, ← c79808] at e_1298_7
  have hr := e_1298_7
  simp only [e_1298_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12062)) := by
  have c79809 := (publicBatchWrapper86_copies2494 a h).2.1
  have c79810 := (publicBatchWrapper86_copies2494 a h).2.2.1
  have c79811 := (publicBatchWrapper86_copies2494 a h).2.2.2.1
  have c79812 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.1
  have c79813 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.1
  have c79814 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_8
  simp only [← c79809, ← c79810, ← c79811] at e_1298_8
  have e_1298_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_9
  simp only [← c79812, ← c79813, k2, ← c79814] at e_1298_9
  have hr := e_1298_9
  simp only [e_1298_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12063)) := by
  have c79815 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.1
  have c79816 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.1
  have c79817 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.1
  have c79818 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79819 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79820 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_10
  simp only [← c79815, ← c79816, ← c79817] at e_1298_10
  have e_1298_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_11
  simp only [← c79818, ← c79819, k2, ← c79820] at e_1298_11
  have hr := e_1298_11
  simp only [e_1298_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12064)) := by
  have c79821 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79822 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79823 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79824 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79825 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79826 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_12
  simp only [← c79821, ← c79822, ← c79823] at e_1298_12
  have e_1298_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_13
  simp only [← c79824, ← c79825, k2, ← c79826] at e_1298_13
  have hr := e_1298_13
  simp only [e_1298_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12065)) := by
  have c79827 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79828 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79829 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79830 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79831 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79832 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_14
  simp only [← c79827, ← c79828, ← c79829] at e_1298_14
  have e_1298_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_15
  simp only [← c79830, ← c79831, k2, ← c79832] at e_1298_15
  have hr := e_1298_15
  simp only [e_1298_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12066)) := by
  have c79833 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79834 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79835 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79836 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79837 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79838 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_16
  simp only [← c79833, ← c79834, ← c79835] at e_1298_16
  have e_1298_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_17
  simp only [← c79836, ← c79837, k2, ← c79838] at e_1298_17
  have hr := e_1298_17
  simp only [e_1298_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12067)) := by
  have c79839 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79840 := (publicBatchWrapper86_copies2495 a h).1
  have c79841 := (publicBatchWrapper86_copies2495 a h).2.1
  have c79842 := (publicBatchWrapper86_copies2495 a h).2.2.1
  have c79843 := (publicBatchWrapper86_copies2495 a h).2.2.2.1
  have c79844 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_18
  simp only [← c79839, ← c79840, ← c79841] at e_1298_18
  have e_1298_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_19
  simp only [← c79842, ← c79843, k2, ← c79844] at e_1298_19
  have hr := e_1298_19
  simp only [e_1298_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12068)) := by
  have c79845 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.1
  have c79846 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.1
  have c79847 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.1
  have c79848 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.1
  have c79849 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.1
  have c79850 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_0
  simp only [← c79845, ← c79846, ← c79847] at e_1299_0
  have e_1299_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_1
  simp only [← c79848, ← c79849, k2, ← c79850] at e_1299_1
  have hr := e_1299_1
  simp only [e_1299_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12069)) := by
  have c79851 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79852 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79853 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79854 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79855 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79856 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_2
  simp only [← c79851, ← c79852, ← c79853] at e_1299_2
  have e_1299_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_3
  simp only [← c79854, ← c79855, k2, ← c79856] at e_1299_3
  have hr := e_1299_3
  simp only [e_1299_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12070)) := by
  have c79857 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79858 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79859 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79860 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79861 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79862 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_4
  simp only [← c79857, ← c79858, ← c79859] at e_1299_4
  have e_1299_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_5
  simp only [← c79860, ← c79861, k2, ← c79862] at e_1299_5
  have hr := e_1299_5
  simp only [e_1299_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12071)) := by
  have c79863 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79864 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79865 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79866 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79867 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79868 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_6
  simp only [← c79863, ← c79864, ← c79865] at e_1299_6
  have e_1299_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_7
  simp only [← c79866, ← c79867, k2, ← c79868] at e_1299_7
  have hr := e_1299_7
  simp only [e_1299_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12072)) := by
  have c79869 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79870 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79871 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79872 := (publicBatchWrapper86_copies2496 a h).1
  have c79873 := (publicBatchWrapper86_copies2496 a h).2.1
  have c79874 := (publicBatchWrapper86_copies2496 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_8
  simp only [← c79869, ← c79870, ← c79871] at e_1299_8
  have e_1299_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_9
  simp only [← c79872, ← c79873, k2, ← c79874] at e_1299_9
  have hr := e_1299_9
  simp only [e_1299_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12073)) := by
  have c79875 := (publicBatchWrapper86_copies2496 a h).2.2.2.1
  have c79876 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.1
  have c79877 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.1
  have c79878 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.1
  have c79879 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.1
  have c79880 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_10
  simp only [← c79875, ← c79876, ← c79877] at e_1299_10
  have e_1299_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_11
  simp only [← c79878, ← c79879, k2, ← c79880] at e_1299_11
  have hr := e_1299_11
  simp only [e_1299_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12074)) := by
  have c79881 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.1
  have c79882 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79883 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79884 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79885 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79886 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_12
  simp only [← c79881, ← c79882, ← c79883] at e_1299_12
  have e_1299_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_13
  simp only [← c79884, ← c79885, k2, ← c79886] at e_1299_13
  have hr := e_1299_13
  simp only [e_1299_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12075)) := by
  have c79887 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79888 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79889 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79890 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79891 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79892 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_14
  simp only [← c79887, ← c79888, ← c79889] at e_1299_14
  have e_1299_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_15
  simp only [← c79890, ← c79891, k2, ← c79892] at e_1299_15
  have hr := e_1299_15
  simp only [e_1299_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12076)) := by
  have c79893 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79894 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79895 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79896 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79897 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79898 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_16
  simp only [← c79893, ← c79894, ← c79895] at e_1299_16
  have e_1299_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_17
  simp only [← c79896, ← c79897, k2, ← c79898] at e_1299_17
  have hr := e_1299_17
  simp only [e_1299_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12077)) := by
  have c79899 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79900 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79901 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79902 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79903 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79904 := (publicBatchWrapper86_copies2497 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_18
  simp only [← c79899, ← c79900, ← c79901] at e_1299_18
  have e_1299_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_19
  simp only [← c79902, ← c79903, k2, ← c79904] at e_1299_19
  have hr := e_1299_19
  simp only [e_1299_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12078)) := by
  have c79905 := (publicBatchWrapper86_copies2497 a h).2.1
  have c79906 := (publicBatchWrapper86_copies2497 a h).2.2.1
  have c79907 := (publicBatchWrapper86_copies2497 a h).2.2.2.1
  have c79908 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.1
  have c79909 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.1
  have c79910 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_0
  simp only [← c79905, ← c79906, ← c79907] at e_1300_0
  have e_1300_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_1
  simp only [← c79908, ← c79909, k2, ← c79910] at e_1300_1
  have hr := e_1300_1
  simp only [e_1300_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12079)) := by
  have c79911 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.1
  have c79912 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.1
  have c79913 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.1
  have c79914 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79915 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79916 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_2
  simp only [← c79911, ← c79912, ← c79913] at e_1300_2
  have e_1300_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_3
  simp only [← c79914, ← c79915, k2, ← c79916] at e_1300_3
  have hr := e_1300_3
  simp only [e_1300_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12080)) := by
  have c79917 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79918 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79919 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79920 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79921 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79922 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_4
  simp only [← c79917, ← c79918, ← c79919] at e_1300_4
  have e_1300_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_5
  simp only [← c79920, ← c79921, k2, ← c79922] at e_1300_5
  have hr := e_1300_5
  simp only [e_1300_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12081)) := by
  have c79923 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79924 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79925 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79926 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79927 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79928 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_6
  simp only [← c79923, ← c79924, ← c79925] at e_1300_6
  have e_1300_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_7
  simp only [← c79926, ← c79927, k2, ← c79928] at e_1300_7
  have hr := e_1300_7
  simp only [e_1300_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12082)) := by
  have c79929 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79930 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79931 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79932 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79933 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79934 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_8
  simp only [← c79929, ← c79930, ← c79931] at e_1300_8
  have e_1300_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_9
  simp only [← c79932, ← c79933, k2, ← c79934] at e_1300_9
  have hr := e_1300_9
  simp only [e_1300_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12083)) := by
  have c79935 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79936 := (publicBatchWrapper86_copies2498 a h).1
  have c79937 := (publicBatchWrapper86_copies2498 a h).2.1
  have c79938 := (publicBatchWrapper86_copies2498 a h).2.2.1
  have c79939 := (publicBatchWrapper86_copies2498 a h).2.2.2.1
  have c79940 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_10
  simp only [← c79935, ← c79936, ← c79937] at e_1300_10
  have e_1300_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_11
  simp only [← c79938, ← c79939, k2, ← c79940] at e_1300_11
  have hr := e_1300_11
  simp only [e_1300_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12084)) := by
  have c79941 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.1
  have c79942 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.1
  have c79943 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.1
  have c79944 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.1
  have c79945 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.1
  have c79946 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_12
  simp only [← c79941, ← c79942, ← c79943] at e_1300_12
  have e_1300_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_13
  simp only [← c79944, ← c79945, k2, ← c79946] at e_1300_13
  have hr := e_1300_13
  simp only [e_1300_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12085)) := by
  have c79947 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79948 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79949 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79950 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79951 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79952 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_14
  simp only [← c79947, ← c79948, ← c79949] at e_1300_14
  have e_1300_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_15
  simp only [← c79950, ← c79951, k2, ← c79952] at e_1300_15
  have hr := e_1300_15
  simp only [e_1300_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12086)) := by
  have c79953 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79954 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79955 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79956 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79957 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79958 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_16
  simp only [← c79953, ← c79954, ← c79955] at e_1300_16
  have e_1300_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_17
  simp only [← c79956, ← c79957, k2, ← c79958] at e_1300_17
  have hr := e_1300_17
  simp only [e_1300_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12087)) := by
  have c79959 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79960 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79961 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79962 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79963 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79964 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_18
  simp only [← c79959, ← c79960, ← c79961] at e_1300_18
  have e_1300_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_19
  simp only [← c79962, ← c79963, k2, ← c79964] at e_1300_19
  have hr := e_1300_19
  simp only [e_1300_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12088)) := by
  have c79965 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79966 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79967 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79968 := (publicBatchWrapper86_copies2499 a h).1
  have c79969 := (publicBatchWrapper86_copies2499 a h).2.1
  have c79970 := (publicBatchWrapper86_copies2499 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_0
  simp only [← c79965, ← c79966, ← c79967] at e_1301_0
  have e_1301_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_1
  simp only [← c79968, ← c79969, k2, ← c79970] at e_1301_1
  have hr := e_1301_1
  simp only [e_1301_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12089)) := by
  have c79971 := (publicBatchWrapper86_copies2499 a h).2.2.2.1
  have c79972 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.1
  have c79973 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.1
  have c79974 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.1
  have c79975 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.1
  have c79976 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_2
  simp only [← c79971, ← c79972, ← c79973] at e_1301_2
  have e_1301_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_3
  simp only [← c79974, ← c79975, k2, ← c79976] at e_1301_3
  have hr := e_1301_3
  simp only [e_1301_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12090)) := by
  have c79977 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.1
  have c79978 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79979 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79980 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79981 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79982 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_4
  simp only [← c79977, ← c79978, ← c79979] at e_1301_4
  have e_1301_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_5
  simp only [← c79980, ← c79981, k2, ← c79982] at e_1301_5
  have hr := e_1301_5
  simp only [e_1301_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12091)) := by
  have c79983 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79984 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79985 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79986 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79987 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79988 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_6
  simp only [← c79983, ← c79984, ← c79985] at e_1301_6
  have e_1301_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_7
  simp only [← c79986, ← c79987, k2, ← c79988] at e_1301_7
  have hr := e_1301_7
  simp only [e_1301_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12092)) := by
  have c79989 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79990 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79991 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79992 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79993 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79994 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_8
  simp only [← c79989, ← c79990, ← c79991] at e_1301_8
  have e_1301_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_9
  simp only [← c79992, ← c79993, k2, ← c79994] at e_1301_9
  have hr := e_1301_9
  simp only [e_1301_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12093)) := by
  have c79995 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79996 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79997 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79998 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79999 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80000 := (publicBatchWrapper86_copies2500 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_10
  simp only [← c79995, ← c79996, ← c79997] at e_1301_10
  have e_1301_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_11
  simp only [← c79998, ← c79999, k2, ← c80000] at e_1301_11
  have hr := e_1301_11
  simp only [e_1301_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12094)) := by
  have c80001 := (publicBatchWrapper86_copies2500 a h).2.1
  have c80002 := (publicBatchWrapper86_copies2500 a h).2.2.1
  have c80003 := (publicBatchWrapper86_copies2500 a h).2.2.2.1
  have c80004 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.1
  have c80005 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.1
  have c80006 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_12
  simp only [← c80001, ← c80002, ← c80003] at e_1301_12
  have e_1301_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_13
  simp only [← c80004, ← c80005, k2, ← c80006] at e_1301_13
  have hr := e_1301_13
  simp only [e_1301_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12095)) := by
  have c80007 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.1
  have c80008 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.1
  have c80009 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.1
  have c80010 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80011 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80012 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_14
  simp only [← c80007, ← c80008, ← c80009] at e_1301_14
  have e_1301_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_15
  simp only [← c80010, ← c80011, k2, ← c80012] at e_1301_15
  have hr := e_1301_15
  simp only [e_1301_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12096)) := by
  have c80013 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80014 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80015 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80016 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80017 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80018 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_16
  simp only [← c80013, ← c80014, ← c80015] at e_1301_16
  have e_1301_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_17
  simp only [← c80016, ← c80017, k2, ← c80018] at e_1301_17
  have hr := e_1301_17
  simp only [e_1301_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12097)) := by
  have c80019 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80020 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80021 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80022 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80023 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80024 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_18
  simp only [← c80019, ← c80020, ← c80021] at e_1301_18
  have e_1301_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_19
  simp only [← c80022, ← c80023, k2, ← c80024] at e_1301_19
  have hr := e_1301_19
  simp only [e_1301_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12098)) := by
  have c80025 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80026 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80027 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80028 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80029 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80030 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_0
  simp only [← c80025, ← c80026, ← c80027] at e_1302_0
  have e_1302_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_1
  simp only [← c80028, ← c80029, k2, ← c80030] at e_1302_1
  have hr := e_1302_1
  simp only [e_1302_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12099)) := by
  have c80031 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80032 := (publicBatchWrapper86_copies2501 a h).1
  have c80033 := (publicBatchWrapper86_copies2501 a h).2.1
  have c80034 := (publicBatchWrapper86_copies2501 a h).2.2.1
  have c80035 := (publicBatchWrapper86_copies2501 a h).2.2.2.1
  have c80036 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_2
  simp only [← c80031, ← c80032, ← c80033] at e_1302_2
  have e_1302_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_3
  simp only [← c80034, ← c80035, k2, ← c80036] at e_1302_3
  have hr := e_1302_3
  simp only [e_1302_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12100)) := by
  have c80037 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.1
  have c80038 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.1
  have c80039 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.1
  have c80040 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.1
  have c80041 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.1
  have c80042 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_4
  simp only [← c80037, ← c80038, ← c80039] at e_1302_4
  have e_1302_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_5
  simp only [← c80040, ← c80041, k2, ← c80042] at e_1302_5
  have hr := e_1302_5
  simp only [e_1302_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12101)) := by
  have c80043 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80044 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80045 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80046 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80047 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80048 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_6
  simp only [← c80043, ← c80044, ← c80045] at e_1302_6
  have e_1302_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_7
  simp only [← c80046, ← c80047, k2, ← c80048] at e_1302_7
  have hr := e_1302_7
  simp only [e_1302_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12102)) := by
  have c80049 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80050 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80051 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80052 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80053 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80054 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_8
  simp only [← c80049, ← c80050, ← c80051] at e_1302_8
  have e_1302_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_9
  simp only [← c80052, ← c80053, k2, ← c80054] at e_1302_9
  have hr := e_1302_9
  simp only [e_1302_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12103)) := by
  have c80055 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80056 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80057 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80058 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80059 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80060 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_10
  simp only [← c80055, ← c80056, ← c80057] at e_1302_10
  have e_1302_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_11
  simp only [← c80058, ← c80059, k2, ← c80060] at e_1302_11
  have hr := e_1302_11
  simp only [e_1302_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12104)) := by
  have c80061 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80062 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80063 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80064 := (publicBatchWrapper86_copies2502 a h).1
  have c80065 := (publicBatchWrapper86_copies2502 a h).2.1
  have c80066 := (publicBatchWrapper86_copies2502 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_12
  simp only [← c80061, ← c80062, ← c80063] at e_1302_12
  have e_1302_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_13
  simp only [← c80064, ← c80065, k2, ← c80066] at e_1302_13
  have hr := e_1302_13
  simp only [e_1302_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12105)) := by
  have c80067 := (publicBatchWrapper86_copies2502 a h).2.2.2.1
  have c80068 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.1
  have c80069 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.1
  have c80070 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.1
  have c80071 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.1
  have c80072 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_14
  simp only [← c80067, ← c80068, ← c80069] at e_1302_14
  have e_1302_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_15
  simp only [← c80070, ← c80071, k2, ← c80072] at e_1302_15
  have hr := e_1302_15
  simp only [e_1302_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12106)) := by
  have c80073 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.1
  have c80074 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80075 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80076 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80077 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80078 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_16
  simp only [← c80073, ← c80074, ← c80075] at e_1302_16
  have e_1302_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_17
  simp only [← c80076, ← c80077, k2, ← c80078] at e_1302_17
  have hr := e_1302_17
  simp only [e_1302_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12107)) := by
  have c80079 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80080 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80081 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80082 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80083 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80084 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_18
  simp only [← c80079, ← c80080, ← c80081] at e_1302_18
  have e_1302_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_19
  simp only [← c80082, ← c80083, k2, ← c80084] at e_1302_19
  have hr := e_1302_19
  simp only [e_1302_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12108)) := by
  have c80085 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80086 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80087 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80088 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80089 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80090 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_0
  simp only [← c80085, ← c80086, ← c80087] at e_1303_0
  have e_1303_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_1
  simp only [← c80088, ← c80089, k2, ← c80090] at e_1303_1
  have hr := e_1303_1
  simp only [e_1303_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12109)) := by
  have c80091 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80092 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80093 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80094 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80095 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80096 := (publicBatchWrapper86_copies2503 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_2
  simp only [← c80091, ← c80092, ← c80093] at e_1303_2
  have e_1303_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_3
  simp only [← c80094, ← c80095, k2, ← c80096] at e_1303_3
  have hr := e_1303_3
  simp only [e_1303_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12110)) := by
  have c80097 := (publicBatchWrapper86_copies2503 a h).2.1
  have c80098 := (publicBatchWrapper86_copies2503 a h).2.2.1
  have c80099 := (publicBatchWrapper86_copies2503 a h).2.2.2.1
  have c80100 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.1
  have c80101 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.1
  have c80102 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_4
  simp only [← c80097, ← c80098, ← c80099] at e_1303_4
  have e_1303_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_5
  simp only [← c80100, ← c80101, k2, ← c80102] at e_1303_5
  have hr := e_1303_5
  simp only [e_1303_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12111)) := by
  have c80103 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.1
  have c80104 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.1
  have c80105 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.1
  have c80106 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80107 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80108 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_6
  simp only [← c80103, ← c80104, ← c80105] at e_1303_6
  have e_1303_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_7
  simp only [← c80106, ← c80107, k2, ← c80108] at e_1303_7
  have hr := e_1303_7
  simp only [e_1303_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12112)) := by
  have c80109 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80110 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80111 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80112 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80113 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80114 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_8
  simp only [← c80109, ← c80110, ← c80111] at e_1303_8
  have e_1303_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_9
  simp only [← c80112, ← c80113, k2, ← c80114] at e_1303_9
  have hr := e_1303_9
  simp only [e_1303_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12113)) := by
  have c80115 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80116 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80117 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80118 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80119 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80120 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_10
  simp only [← c80115, ← c80116, ← c80117] at e_1303_10
  have e_1303_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_11
  simp only [← c80118, ← c80119, k2, ← c80120] at e_1303_11
  have hr := e_1303_11
  simp only [e_1303_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12114)) := by
  have c80121 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80122 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80123 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80124 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80125 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80126 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_12
  simp only [← c80121, ← c80122, ← c80123] at e_1303_12
  have e_1303_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_13
  simp only [← c80124, ← c80125, k2, ← c80126] at e_1303_13
  have hr := e_1303_13
  simp only [e_1303_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12115)) := by
  have c80127 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80128 := (publicBatchWrapper86_copies2504 a h).1
  have c80129 := (publicBatchWrapper86_copies2504 a h).2.1
  have c80130 := (publicBatchWrapper86_copies2504 a h).2.2.1
  have c80131 := (publicBatchWrapper86_copies2504 a h).2.2.2.1
  have c80132 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_14
  simp only [← c80127, ← c80128, ← c80129] at e_1303_14
  have e_1303_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_15
  simp only [← c80130, ← c80131, k2, ← c80132] at e_1303_15
  have hr := e_1303_15
  simp only [e_1303_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12116)) := by
  have c80133 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.1
  have c80134 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.1
  have c80135 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.1
  have c80136 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.1
  have c80137 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.1
  have c80138 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_16
  simp only [← c80133, ← c80134, ← c80135] at e_1303_16
  have e_1303_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_17
  simp only [← c80136, ← c80137, k2, ← c80138] at e_1303_17
  have hr := e_1303_17
  simp only [e_1303_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12117)) := by
  have c80139 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80140 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80141 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80142 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80143 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80144 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_18
  simp only [← c80139, ← c80140, ← c80141] at e_1303_18
  have e_1303_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_19
  simp only [← c80142, ← c80143, k2, ← c80144] at e_1303_19
  have hr := e_1303_19
  simp only [e_1303_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12118)) := by
  have c80145 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80146 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80147 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80148 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80149 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80150 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_0
  simp only [← c80145, ← c80146, ← c80147] at e_1304_0
  have e_1304_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_1
  simp only [← c80148, ← c80149, k2, ← c80150] at e_1304_1
  have hr := e_1304_1
  simp only [e_1304_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12119)) := by
  have c80151 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80152 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80153 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80154 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80155 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80156 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_2
  simp only [← c80151, ← c80152, ← c80153] at e_1304_2
  have e_1304_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_3
  simp only [← c80154, ← c80155, k2, ← c80156] at e_1304_3
  have hr := e_1304_3
  simp only [e_1304_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12120)) := by
  have c80157 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80158 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80159 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80160 := (publicBatchWrapper86_copies2505 a h).1
  have c80161 := (publicBatchWrapper86_copies2505 a h).2.1
  have c80162 := (publicBatchWrapper86_copies2505 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_4
  simp only [← c80157, ← c80158, ← c80159] at e_1304_4
  have e_1304_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_5
  simp only [← c80160, ← c80161, k2, ← c80162] at e_1304_5
  have hr := e_1304_5
  simp only [e_1304_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12121)) := by
  have c80163 := (publicBatchWrapper86_copies2505 a h).2.2.2.1
  have c80164 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.1
  have c80165 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.1
  have c80166 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.1
  have c80167 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.1
  have c80168 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_6
  simp only [← c80163, ← c80164, ← c80165] at e_1304_6
  have e_1304_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_7
  simp only [← c80166, ← c80167, k2, ← c80168] at e_1304_7
  have hr := e_1304_7
  simp only [e_1304_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12122)) := by
  have c80169 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.1
  have c80170 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80171 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80172 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80173 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80174 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_8
  simp only [← c80169, ← c80170, ← c80171] at e_1304_8
  have e_1304_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_9
  simp only [← c80172, ← c80173, k2, ← c80174] at e_1304_9
  have hr := e_1304_9
  simp only [e_1304_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12123)) := by
  have c80175 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80176 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80177 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80178 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80179 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80180 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_10
  simp only [← c80175, ← c80176, ← c80177] at e_1304_10
  have e_1304_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_11
  simp only [← c80178, ← c80179, k2, ← c80180] at e_1304_11
  have hr := e_1304_11
  simp only [e_1304_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12124)) := by
  have c80181 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80182 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80183 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80184 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80185 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80186 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_12
  simp only [← c80181, ← c80182, ← c80183] at e_1304_12
  have e_1304_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_13
  simp only [← c80184, ← c80185, k2, ← c80186] at e_1304_13
  have hr := e_1304_13
  simp only [e_1304_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12125)) := by
  have c80187 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80188 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80189 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80190 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80191 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80192 := (publicBatchWrapper86_copies2506 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_14
  simp only [← c80187, ← c80188, ← c80189] at e_1304_14
  have e_1304_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_15
  simp only [← c80190, ← c80191, k2, ← c80192] at e_1304_15
  have hr := e_1304_15
  simp only [e_1304_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12126)) := by
  have c80193 := (publicBatchWrapper86_copies2506 a h).2.1
  have c80194 := (publicBatchWrapper86_copies2506 a h).2.2.1
  have c80195 := (publicBatchWrapper86_copies2506 a h).2.2.2.1
  have c80196 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.1
  have c80197 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.1
  have c80198 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_16
  simp only [← c80193, ← c80194, ← c80195] at e_1304_16
  have e_1304_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_17
  simp only [← c80196, ← c80197, k2, ← c80198] at e_1304_17
  have hr := e_1304_17
  simp only [e_1304_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12127)) := by
  have c80199 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.1
  have c80200 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.1
  have c80201 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.1
  have c80202 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80203 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80204 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_18
  simp only [← c80199, ← c80200, ← c80201] at e_1304_18
  have e_1304_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_19
  simp only [← c80202, ← c80203, k2, ← c80204] at e_1304_19
  have hr := e_1304_19
  simp only [e_1304_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12128)) := by
  have c80205 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80206 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80207 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80208 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80209 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80210 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_0
  simp only [← c80205, ← c80206, ← c80207] at e_1305_0
  have e_1305_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_1
  simp only [← c80208, ← c80209, k2, ← c80210] at e_1305_1
  have hr := e_1305_1
  simp only [e_1305_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12129)) := by
  have c80211 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80212 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80213 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80214 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80215 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80216 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_2
  simp only [← c80211, ← c80212, ← c80213] at e_1305_2
  have e_1305_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_3
  simp only [← c80214, ← c80215, k2, ← c80216] at e_1305_3
  have hr := e_1305_3
  simp only [e_1305_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12130)) := by
  have c80217 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80218 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80219 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80220 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80221 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80222 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_4
  simp only [← c80217, ← c80218, ← c80219] at e_1305_4
  have e_1305_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_5
  simp only [← c80220, ← c80221, k2, ← c80222] at e_1305_5
  have hr := e_1305_5
  simp only [e_1305_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12131)) := by
  have c80223 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80224 := (publicBatchWrapper86_copies2507 a h).1
  have c80225 := (publicBatchWrapper86_copies2507 a h).2.1
  have c80226 := (publicBatchWrapper86_copies2507 a h).2.2.1
  have c80227 := (publicBatchWrapper86_copies2507 a h).2.2.2.1
  have c80228 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_6
  simp only [← c80223, ← c80224, ← c80225] at e_1305_6
  have e_1305_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_7
  simp only [← c80226, ← c80227, k2, ← c80228] at e_1305_7
  have hr := e_1305_7
  simp only [e_1305_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12132)) := by
  have c80229 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.1
  have c80230 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.1
  have c80231 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.1
  have c80232 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.1
  have c80233 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.1
  have c80234 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_8
  simp only [← c80229, ← c80230, ← c80231] at e_1305_8
  have e_1305_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_9
  simp only [← c80232, ← c80233, k2, ← c80234] at e_1305_9
  have hr := e_1305_9
  simp only [e_1305_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12133)) := by
  have c80235 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80236 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80237 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80238 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80239 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80240 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_10
  simp only [← c80235, ← c80236, ← c80237] at e_1305_10
  have e_1305_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_11
  simp only [← c80238, ← c80239, k2, ← c80240] at e_1305_11
  have hr := e_1305_11
  simp only [e_1305_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12134)) := by
  have c80241 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80242 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80243 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80244 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80245 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80246 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_12
  simp only [← c80241, ← c80242, ← c80243] at e_1305_12
  have e_1305_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_13
  simp only [← c80244, ← c80245, k2, ← c80246] at e_1305_13
  have hr := e_1305_13
  simp only [e_1305_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12135)) := by
  have c80247 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80248 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80249 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80250 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80251 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80252 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_14
  simp only [← c80247, ← c80248, ← c80249] at e_1305_14
  have e_1305_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_15
  simp only [← c80250, ← c80251, k2, ← c80252] at e_1305_15
  have hr := e_1305_15
  simp only [e_1305_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12136)) := by
  have c80253 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80254 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80255 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80256 := (publicBatchWrapper86_copies2508 a h).1
  have c80257 := (publicBatchWrapper86_copies2508 a h).2.1
  have c80258 := (publicBatchWrapper86_copies2508 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_16
  simp only [← c80253, ← c80254, ← c80255] at e_1305_16
  have e_1305_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_17
  simp only [← c80256, ← c80257, k2, ← c80258] at e_1305_17
  have hr := e_1305_17
  simp only [e_1305_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12137)) := by
  have c80259 := (publicBatchWrapper86_copies2508 a h).2.2.2.1
  have c80260 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.1
  have c80261 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.1
  have c80262 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.1
  have c80263 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.1
  have c80264 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_18
  simp only [← c80259, ← c80260, ← c80261] at e_1305_18
  have e_1305_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_19
  simp only [← c80262, ← c80263, k2, ← c80264] at e_1305_19
  have hr := e_1305_19
  simp only [e_1305_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12138)) := by
  have c80265 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.1
  have c80266 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80267 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80268 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80269 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80270 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_0
  simp only [← c80265, ← c80266, ← c80267] at e_1306_0
  have e_1306_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_1
  simp only [← c80268, ← c80269, k2, ← c80270] at e_1306_1
  have hr := e_1306_1
  simp only [e_1306_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12139)) := by
  have c80271 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80272 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80273 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80274 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80275 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80276 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_2
  simp only [← c80271, ← c80272, ← c80273] at e_1306_2
  have e_1306_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_3
  simp only [← c80274, ← c80275, k2, ← c80276] at e_1306_3
  have hr := e_1306_3
  simp only [e_1306_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12140)) := by
  have c80277 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80278 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80279 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80280 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80281 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80282 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_4
  simp only [← c80277, ← c80278, ← c80279] at e_1306_4
  have e_1306_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_5
  simp only [← c80280, ← c80281, k2, ← c80282] at e_1306_5
  have hr := e_1306_5
  simp only [e_1306_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12141)) := by
  have c80283 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80284 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80285 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80286 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80287 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80288 := (publicBatchWrapper86_copies2509 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_6
  simp only [← c80283, ← c80284, ← c80285] at e_1306_6
  have e_1306_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_7
  simp only [← c80286, ← c80287, k2, ← c80288] at e_1306_7
  have hr := e_1306_7
  simp only [e_1306_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12142)) := by
  have c80289 := (publicBatchWrapper86_copies2509 a h).2.1
  have c80290 := (publicBatchWrapper86_copies2509 a h).2.2.1
  have c80291 := (publicBatchWrapper86_copies2509 a h).2.2.2.1
  have c80292 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.1
  have c80293 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.1
  have c80294 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_8
  simp only [← c80289, ← c80290, ← c80291] at e_1306_8
  have e_1306_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_9
  simp only [← c80292, ← c80293, k2, ← c80294] at e_1306_9
  have hr := e_1306_9
  simp only [e_1306_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12143)) := by
  have c80295 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.1
  have c80296 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.1
  have c80297 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.1
  have c80298 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80299 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80300 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_10
  simp only [← c80295, ← c80296, ← c80297] at e_1306_10
  have e_1306_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_11
  simp only [← c80298, ← c80299, k2, ← c80300] at e_1306_11
  have hr := e_1306_11
  simp only [e_1306_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12144)) := by
  have c80301 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80302 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80303 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80304 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80305 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80306 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_12
  simp only [← c80301, ← c80302, ← c80303] at e_1306_12
  have e_1306_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_13
  simp only [← c80304, ← c80305, k2, ← c80306] at e_1306_13
  have hr := e_1306_13
  simp only [e_1306_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12145)) := by
  have c80307 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80308 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80309 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80310 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80311 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80312 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_14
  simp only [← c80307, ← c80308, ← c80309] at e_1306_14
  have e_1306_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_15
  simp only [← c80310, ← c80311, k2, ← c80312] at e_1306_15
  have hr := e_1306_15
  simp only [e_1306_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12146)) := by
  have c80313 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80314 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80315 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80316 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80317 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80318 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_16
  simp only [← c80313, ← c80314, ← c80315] at e_1306_16
  have e_1306_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_17
  simp only [← c80316, ← c80317, k2, ← c80318] at e_1306_17
  have hr := e_1306_17
  simp only [e_1306_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12147)) := by
  have c80319 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80320 := (publicBatchWrapper86_copies2510 a h).1
  have c80321 := (publicBatchWrapper86_copies2510 a h).2.1
  have c80322 := (publicBatchWrapper86_copies2510 a h).2.2.1
  have c80323 := (publicBatchWrapper86_copies2510 a h).2.2.2.1
  have c80324 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_18
  simp only [← c80319, ← c80320, ← c80321] at e_1306_18
  have e_1306_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_19
  simp only [← c80322, ← c80323, k2, ← c80324] at e_1306_19
  have hr := e_1306_19
  simp only [e_1306_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12148)) := by
  have c80325 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.1
  have c80326 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.1
  have c80327 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.1
  have c80328 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.1
  have c80329 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.1
  have c80330 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_0
  simp only [← c80325, ← c80326, ← c80327] at e_1307_0
  have e_1307_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_1
  simp only [← c80328, ← c80329, k2, ← c80330] at e_1307_1
  have hr := e_1307_1
  simp only [e_1307_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12149)) := by
  have c80331 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80332 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80333 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80334 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80335 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80336 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_2
  simp only [← c80331, ← c80332, ← c80333] at e_1307_2
  have e_1307_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_3
  simp only [← c80334, ← c80335, k2, ← c80336] at e_1307_3
  have hr := e_1307_3
  simp only [e_1307_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12150)) := by
  have c80337 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80338 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80339 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80340 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80341 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80342 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_4
  simp only [← c80337, ← c80338, ← c80339] at e_1307_4
  have e_1307_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_5
  simp only [← c80340, ← c80341, k2, ← c80342] at e_1307_5
  have hr := e_1307_5
  simp only [e_1307_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12151)) := by
  have c80343 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80344 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80345 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80346 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80347 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80348 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_6
  simp only [← c80343, ← c80344, ← c80345] at e_1307_6
  have e_1307_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_7
  simp only [← c80346, ← c80347, k2, ← c80348] at e_1307_7
  have hr := e_1307_7
  simp only [e_1307_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12152)) := by
  have c80349 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80350 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80351 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80352 := (publicBatchWrapper86_copies2511 a h).1
  have c80353 := (publicBatchWrapper86_copies2511 a h).2.1
  have c80354 := (publicBatchWrapper86_copies2511 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_8
  simp only [← c80349, ← c80350, ← c80351] at e_1307_8
  have e_1307_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_9
  simp only [← c80352, ← c80353, k2, ← c80354] at e_1307_9
  have hr := e_1307_9
  simp only [e_1307_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12153)) := by
  have c80355 := (publicBatchWrapper86_copies2511 a h).2.2.2.1
  have c80356 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.1
  have c80357 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.1
  have c80358 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.1
  have c80359 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.1
  have c80360 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_10
  simp only [← c80355, ← c80356, ← c80357] at e_1307_10
  have e_1307_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_11
  simp only [← c80358, ← c80359, k2, ← c80360] at e_1307_11
  have hr := e_1307_11
  simp only [e_1307_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12154)) := by
  have c80361 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.1
  have c80362 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80363 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80364 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80365 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80366 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_12
  simp only [← c80361, ← c80362, ← c80363] at e_1307_12
  have e_1307_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_13
  simp only [← c80364, ← c80365, k2, ← c80366] at e_1307_13
  have hr := e_1307_13
  simp only [e_1307_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12155)) := by
  have c80367 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80368 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80369 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80370 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80371 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80372 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_14
  simp only [← c80367, ← c80368, ← c80369] at e_1307_14
  have e_1307_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_15
  simp only [← c80370, ← c80371, k2, ← c80372] at e_1307_15
  have hr := e_1307_15
  simp only [e_1307_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12156)) := by
  have c80373 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80374 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80375 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80376 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80377 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80378 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_16
  simp only [← c80373, ← c80374, ← c80375] at e_1307_16
  have e_1307_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_17
  simp only [← c80376, ← c80377, k2, ← c80378] at e_1307_17
  have hr := e_1307_17
  simp only [e_1307_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12157)) := by
  have c80379 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80380 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80381 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80382 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80383 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80384 := (publicBatchWrapper86_copies2512 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_18
  simp only [← c80379, ← c80380, ← c80381] at e_1307_18
  have e_1307_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_19
  simp only [← c80382, ← c80383, k2, ← c80384] at e_1307_19
  have hr := e_1307_19
  simp only [e_1307_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12158)) := by
  have c80385 := (publicBatchWrapper86_copies2512 a h).2.1
  have c80386 := (publicBatchWrapper86_copies2512 a h).2.2.1
  have c80387 := (publicBatchWrapper86_copies2512 a h).2.2.2.1
  have c80388 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.1
  have c80389 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.1
  have c80390 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_0
  simp only [← c80385, ← c80386, ← c80387] at e_1308_0
  have e_1308_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_1
  simp only [← c80388, ← c80389, k2, ← c80390] at e_1308_1
  have hr := e_1308_1
  simp only [e_1308_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12159)) := by
  have c80391 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.1
  have c80392 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.1
  have c80393 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.1
  have c80394 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80395 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80396 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_2
  simp only [← c80391, ← c80392, ← c80393] at e_1308_2
  have e_1308_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_3
  simp only [← c80394, ← c80395, k2, ← c80396] at e_1308_3
  have hr := e_1308_3
  simp only [e_1308_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12160)) := by
  have c80397 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80398 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80399 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80400 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80401 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80402 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_4
  simp only [← c80397, ← c80398, ← c80399] at e_1308_4
  have e_1308_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_5
  simp only [← c80400, ← c80401, k2, ← c80402] at e_1308_5
  have hr := e_1308_5
  simp only [e_1308_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12161)) := by
  have c80403 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80404 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80405 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80406 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80407 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80408 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_6
  simp only [← c80403, ← c80404, ← c80405] at e_1308_6
  have e_1308_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_7
  simp only [← c80406, ← c80407, k2, ← c80408] at e_1308_7
  have hr := e_1308_7
  simp only [e_1308_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12162)) := by
  have c80409 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80410 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80411 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80412 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80413 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80414 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_8
  simp only [← c80409, ← c80410, ← c80411] at e_1308_8
  have e_1308_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_9
  simp only [← c80412, ← c80413, k2, ← c80414] at e_1308_9
  have hr := e_1308_9
  simp only [e_1308_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12163)) := by
  have c80415 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80416 := (publicBatchWrapper86_copies2513 a h).1
  have c80417 := (publicBatchWrapper86_copies2513 a h).2.1
  have c80418 := (publicBatchWrapper86_copies2513 a h).2.2.1
  have c80419 := (publicBatchWrapper86_copies2513 a h).2.2.2.1
  have c80420 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_10
  simp only [← c80415, ← c80416, ← c80417] at e_1308_10
  have e_1308_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_11
  simp only [← c80418, ← c80419, k2, ← c80420] at e_1308_11
  have hr := e_1308_11
  simp only [e_1308_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12164)) := by
  have c80421 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.1
  have c80422 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.1
  have c80423 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.1
  have c80424 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.1
  have c80425 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.1
  have c80426 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_12
  simp only [← c80421, ← c80422, ← c80423] at e_1308_12
  have e_1308_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_13
  simp only [← c80424, ← c80425, k2, ← c80426] at e_1308_13
  have hr := e_1308_13
  simp only [e_1308_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12165)) := by
  have c80427 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80428 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80429 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80430 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80431 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80432 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_14
  simp only [← c80427, ← c80428, ← c80429] at e_1308_14
  have e_1308_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_15
  simp only [← c80430, ← c80431, k2, ← c80432] at e_1308_15
  have hr := e_1308_15
  simp only [e_1308_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12166)) := by
  have c80433 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80434 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80435 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80436 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80437 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80438 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_16
  simp only [← c80433, ← c80434, ← c80435] at e_1308_16
  have e_1308_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_17
  simp only [← c80436, ← c80437, k2, ← c80438] at e_1308_17
  have hr := e_1308_17
  simp only [e_1308_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12167)) := by
  have c80439 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80440 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80441 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80442 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80443 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80444 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_18
  simp only [← c80439, ← c80440, ← c80441] at e_1308_18
  have e_1308_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_19
  simp only [← c80442, ← c80443, k2, ← c80444] at e_1308_19
  have hr := e_1308_19
  simp only [e_1308_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12168)) := by
  have c80445 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80446 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80447 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80448 := (publicBatchWrapper86_copies2514 a h).1
  have c80449 := (publicBatchWrapper86_copies2514 a h).2.1
  have c80450 := (publicBatchWrapper86_copies2514 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_0
  simp only [← c80445, ← c80446, ← c80447] at e_1309_0
  have e_1309_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_1
  simp only [← c80448, ← c80449, k2, ← c80450] at e_1309_1
  have hr := e_1309_1
  simp only [e_1309_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12169)) := by
  have c80451 := (publicBatchWrapper86_copies2514 a h).2.2.2.1
  have c80452 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.1
  have c80453 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.1
  have c80454 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.1
  have c80455 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.1
  have c80456 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_2
  simp only [← c80451, ← c80452, ← c80453] at e_1309_2
  have e_1309_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_3
  simp only [← c80454, ← c80455, k2, ← c80456] at e_1309_3
  have hr := e_1309_3
  simp only [e_1309_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12210)) := by
  have c80457 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.1
  have c80458 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80459 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80460 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80461 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80462 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_4
  simp only [← c80457, ← c80458, ← c80459] at e_1309_4
  have e_1309_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_5
  simp only [← c80460, ← c80461, k2, ← c80462] at e_1309_5
  have hr := e_1309_5
  simp only [e_1309_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12211)) := by
  have c80463 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80464 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80465 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80466 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80467 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80468 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_6
  simp only [← c80463, ← c80464, ← c80465] at e_1309_6
  have e_1309_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_7
  simp only [← c80466, ← c80467, k2, ← c80468] at e_1309_7
  have hr := e_1309_7
  simp only [e_1309_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12212)) := by
  have c80469 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80470 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80471 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80472 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80473 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80474 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_8
  simp only [← c80469, ← c80470, ← c80471] at e_1309_8
  have e_1309_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_9
  simp only [← c80472, ← c80473, k2, ← c80474] at e_1309_9
  have hr := e_1309_9
  simp only [e_1309_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12213)) := by
  have c80475 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80476 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80477 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80478 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80479 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80480 := (publicBatchWrapper86_copies2515 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_10
  simp only [← c80475, ← c80476, ← c80477] at e_1309_10
  have e_1309_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_11
  simp only [← c80478, ← c80479, k2, ← c80480] at e_1309_11
  have hr := e_1309_11
  simp only [e_1309_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12214)) := by
  have c80481 := (publicBatchWrapper86_copies2515 a h).2.1
  have c80482 := (publicBatchWrapper86_copies2515 a h).2.2.1
  have c80483 := (publicBatchWrapper86_copies2515 a h).2.2.2.1
  have c80484 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.1
  have c80485 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.1
  have c80486 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_12
  simp only [← c80481, ← c80482, ← c80483] at e_1309_12
  have e_1309_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_13
  simp only [← c80484, ← c80485, k2, ← c80486] at e_1309_13
  have hr := e_1309_13
  simp only [e_1309_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12215)) := by
  have c80487 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.1
  have c80488 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.1
  have c80489 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.1
  have c80490 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80491 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80492 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_14
  simp only [← c80487, ← c80488, ← c80489] at e_1309_14
  have e_1309_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_15
  simp only [← c80490, ← c80491, k2, ← c80492] at e_1309_15
  have hr := e_1309_15
  simp only [e_1309_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12216)) := by
  have c80493 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80494 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80495 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80496 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80497 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80498 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_16
  simp only [← c80493, ← c80494, ← c80495] at e_1309_16
  have e_1309_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_17
  simp only [← c80496, ← c80497, k2, ← c80498] at e_1309_17
  have hr := e_1309_17
  simp only [e_1309_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12217)) := by
  have c80499 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80500 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80501 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80502 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80503 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80504 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_18
  simp only [← c80499, ← c80500, ← c80501] at e_1309_18
  have e_1309_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_19
  simp only [← c80502, ← c80503, k2, ← c80504] at e_1309_19
  have hr := e_1309_19
  simp only [e_1309_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12218)) := by
  have c80505 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80506 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80507 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80508 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80509 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80510 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_0
  simp only [← c80505, ← c80506, ← c80507] at e_1310_0
  have e_1310_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_1
  simp only [← c80508, ← c80509, k2, ← c80510] at e_1310_1
  have hr := e_1310_1
  simp only [e_1310_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12219)) := by
  have c80511 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80512 := (publicBatchWrapper86_copies2516 a h).1
  have c80513 := (publicBatchWrapper86_copies2516 a h).2.1
  have c80514 := (publicBatchWrapper86_copies2516 a h).2.2.1
  have c80515 := (publicBatchWrapper86_copies2516 a h).2.2.2.1
  have c80516 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_2
  simp only [← c80511, ← c80512, ← c80513] at e_1310_2
  have e_1310_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_3
  simp only [← c80514, ← c80515, k2, ← c80516] at e_1310_3
  have hr := e_1310_3
  simp only [e_1310_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12220)) := by
  have c80517 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.1
  have c80518 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.1
  have c80519 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.1
  have c80520 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.1
  have c80521 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.1
  have c80522 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_4
  simp only [← c80517, ← c80518, ← c80519] at e_1310_4
  have e_1310_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_5
  simp only [← c80520, ← c80521, k2, ← c80522] at e_1310_5
  have hr := e_1310_5
  simp only [e_1310_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12221)) := by
  have c80523 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80524 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80525 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80526 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80527 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80528 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_6
  simp only [← c80523, ← c80524, ← c80525] at e_1310_6
  have e_1310_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_7
  simp only [← c80526, ← c80527, k2, ← c80528] at e_1310_7
  have hr := e_1310_7
  simp only [e_1310_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12222)) := by
  have c80529 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80530 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80531 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80532 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80533 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80534 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_8
  simp only [← c80529, ← c80530, ← c80531] at e_1310_8
  have e_1310_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_9
  simp only [← c80532, ← c80533, k2, ← c80534] at e_1310_9
  have hr := e_1310_9
  simp only [e_1310_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12223)) := by
  have c80535 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80536 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80537 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80538 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80539 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80540 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_10
  simp only [← c80535, ← c80536, ← c80537] at e_1310_10
  have e_1310_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_11
  simp only [← c80538, ← c80539, k2, ← c80540] at e_1310_11
  have hr := e_1310_11
  simp only [e_1310_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12224)) := by
  have c80541 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80542 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80543 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80544 := (publicBatchWrapper86_copies2517 a h).1
  have c80545 := (publicBatchWrapper86_copies2517 a h).2.1
  have c80546 := (publicBatchWrapper86_copies2517 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_12
  simp only [← c80541, ← c80542, ← c80543] at e_1310_12
  have e_1310_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_13
  simp only [← c80544, ← c80545, k2, ← c80546] at e_1310_13
  have hr := e_1310_13
  simp only [e_1310_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12225)) := by
  have c80547 := (publicBatchWrapper86_copies2517 a h).2.2.2.1
  have c80548 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.1
  have c80549 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.1
  have c80550 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.1
  have c80551 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.1
  have c80552 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_14
  simp only [← c80547, ← c80548, ← c80549] at e_1310_14
  have e_1310_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_15
  simp only [← c80550, ← c80551, k2, ← c80552] at e_1310_15
  have hr := e_1310_15
  simp only [e_1310_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12226)) := by
  have c80553 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.1
  have c80554 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80555 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80556 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80557 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80558 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_16
  simp only [← c80553, ← c80554, ← c80555] at e_1310_16
  have e_1310_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_17
  simp only [← c80556, ← c80557, k2, ← c80558] at e_1310_17
  have hr := e_1310_17
  simp only [e_1310_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12227)) := by
  have c80559 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80560 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80561 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80562 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80563 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80564 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_18
  simp only [← c80559, ← c80560, ← c80561] at e_1310_18
  have e_1310_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_19
  simp only [← c80562, ← c80563, k2, ← c80564] at e_1310_19
  have hr := e_1310_19
  simp only [e_1310_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12228)) := by
  have c80565 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80566 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80567 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80568 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80569 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80570 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_0
  simp only [← c80565, ← c80566, ← c80567] at e_1311_0
  have e_1311_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_1
  simp only [← c80568, ← c80569, k2, ← c80570] at e_1311_1
  have hr := e_1311_1
  simp only [e_1311_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12229)) := by
  have c80571 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80572 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80573 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80574 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80575 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80576 := (publicBatchWrapper86_copies2518 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_2
  simp only [← c80571, ← c80572, ← c80573] at e_1311_2
  have e_1311_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_3
  simp only [← c80574, ← c80575, k2, ← c80576] at e_1311_3
  have hr := e_1311_3
  simp only [e_1311_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12230)) := by
  have c80577 := (publicBatchWrapper86_copies2518 a h).2.1
  have c80578 := (publicBatchWrapper86_copies2518 a h).2.2.1
  have c80579 := (publicBatchWrapper86_copies2518 a h).2.2.2.1
  have c80580 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.1
  have c80581 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.1
  have c80582 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_4
  simp only [← c80577, ← c80578, ← c80579] at e_1311_4
  have e_1311_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_5
  simp only [← c80580, ← c80581, k2, ← c80582] at e_1311_5
  have hr := e_1311_5
  simp only [e_1311_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12231)) := by
  have c80583 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.1
  have c80584 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.1
  have c80585 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.1
  have c80586 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80587 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80588 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_6
  simp only [← c80583, ← c80584, ← c80585] at e_1311_6
  have e_1311_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_7
  simp only [← c80586, ← c80587, k2, ← c80588] at e_1311_7
  have hr := e_1311_7
  simp only [e_1311_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12232)) := by
  have c80589 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80590 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80591 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80592 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80593 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80594 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_8
  simp only [← c80589, ← c80590, ← c80591] at e_1311_8
  have e_1311_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_9
  simp only [← c80592, ← c80593, k2, ← c80594] at e_1311_9
  have hr := e_1311_9
  simp only [e_1311_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12233)) := by
  have c80595 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80596 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80597 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80598 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80599 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80600 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_10
  simp only [← c80595, ← c80596, ← c80597] at e_1311_10
  have e_1311_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_11
  simp only [← c80598, ← c80599, k2, ← c80600] at e_1311_11
  have hr := e_1311_11
  simp only [e_1311_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12234)) := by
  have c80601 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80602 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80603 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80604 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80605 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80606 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_12
  simp only [← c80601, ← c80602, ← c80603] at e_1311_12
  have e_1311_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_13
  simp only [← c80604, ← c80605, k2, ← c80606] at e_1311_13
  have hr := e_1311_13
  simp only [e_1311_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12235)) := by
  have c80607 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80608 := (publicBatchWrapper86_copies2519 a h).1
  have c80609 := (publicBatchWrapper86_copies2519 a h).2.1
  have c80610 := (publicBatchWrapper86_copies2519 a h).2.2.1
  have c80611 := (publicBatchWrapper86_copies2519 a h).2.2.2.1
  have c80612 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_14
  simp only [← c80607, ← c80608, ← c80609] at e_1311_14
  have e_1311_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_15
  simp only [← c80610, ← c80611, k2, ← c80612] at e_1311_15
  have hr := e_1311_15
  simp only [e_1311_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12236)) := by
  have c80613 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.1
  have c80614 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.1
  have c80615 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.1
  have c80616 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.1
  have c80617 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.1
  have c80618 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_16
  simp only [← c80613, ← c80614, ← c80615] at e_1311_16
  have e_1311_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_17
  simp only [← c80616, ← c80617, k2, ← c80618] at e_1311_17
  have hr := e_1311_17
  simp only [e_1311_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12237)) := by
  have c80619 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80620 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80621 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80622 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80623 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80624 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_18
  simp only [← c80619, ← c80620, ← c80621] at e_1311_18
  have e_1311_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_19
  simp only [← c80622, ← c80623, k2, ← c80624] at e_1311_19
  have hr := e_1311_19
  simp only [e_1311_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12238)) := by
  have c80625 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80626 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80627 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80628 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80629 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80630 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_0
  simp only [← c80625, ← c80626, ← c80627] at e_1312_0
  have e_1312_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_1
  simp only [← c80628, ← c80629, k2, ← c80630] at e_1312_1
  have hr := e_1312_1
  simp only [e_1312_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12239)) := by
  have c80631 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80632 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80633 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80634 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80635 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80636 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_2
  simp only [← c80631, ← c80632, ← c80633] at e_1312_2
  have e_1312_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_3
  simp only [← c80634, ← c80635, k2, ← c80636] at e_1312_3
  have hr := e_1312_3
  simp only [e_1312_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12240)) := by
  have c80637 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80638 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80639 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80640 := (publicBatchWrapper86_copies2520 a h).1
  have c80641 := (publicBatchWrapper86_copies2520 a h).2.1
  have c80642 := (publicBatchWrapper86_copies2520 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_4
  simp only [← c80637, ← c80638, ← c80639] at e_1312_4
  have e_1312_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_5
  simp only [← c80640, ← c80641, k2, ← c80642] at e_1312_5
  have hr := e_1312_5
  simp only [e_1312_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12241)) := by
  have c80643 := (publicBatchWrapper86_copies2520 a h).2.2.2.1
  have c80644 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.1
  have c80645 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.1
  have c80646 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.1
  have c80647 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.1
  have c80648 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_6
  simp only [← c80643, ← c80644, ← c80645] at e_1312_6
  have e_1312_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_7
  simp only [← c80646, ← c80647, k2, ← c80648] at e_1312_7
  have hr := e_1312_7
  simp only [e_1312_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12242)) := by
  have c80649 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.1
  have c80650 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80651 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80652 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80653 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80654 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_8
  simp only [← c80649, ← c80650, ← c80651] at e_1312_8
  have e_1312_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_9
  simp only [← c80652, ← c80653, k2, ← c80654] at e_1312_9
  have hr := e_1312_9
  simp only [e_1312_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12243)) := by
  have c80655 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80656 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80657 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80658 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80659 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80660 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_10
  simp only [← c80655, ← c80656, ← c80657] at e_1312_10
  have e_1312_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_11
  simp only [← c80658, ← c80659, k2, ← c80660] at e_1312_11
  have hr := e_1312_11
  simp only [e_1312_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12244)) := by
  have c80661 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80662 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80663 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80664 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80665 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80666 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_12
  simp only [← c80661, ← c80662, ← c80663] at e_1312_12
  have e_1312_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_13
  simp only [← c80664, ← c80665, k2, ← c80666] at e_1312_13
  have hr := e_1312_13
  simp only [e_1312_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12245)) := by
  have c80667 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80668 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80669 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80670 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80671 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80672 := (publicBatchWrapper86_copies2521 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_14
  simp only [← c80667, ← c80668, ← c80669] at e_1312_14
  have e_1312_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_15
  simp only [← c80670, ← c80671, k2, ← c80672] at e_1312_15
  have hr := e_1312_15
  simp only [e_1312_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12246)) := by
  have c80673 := (publicBatchWrapper86_copies2521 a h).2.1
  have c80674 := (publicBatchWrapper86_copies2521 a h).2.2.1
  have c80675 := (publicBatchWrapper86_copies2521 a h).2.2.2.1
  have c80676 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.1
  have c80677 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.1
  have c80678 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_16
  simp only [← c80673, ← c80674, ← c80675] at e_1312_16
  have e_1312_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_17
  simp only [← c80676, ← c80677, k2, ← c80678] at e_1312_17
  have hr := e_1312_17
  simp only [e_1312_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12247)) := by
  have c80679 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.1
  have c80680 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.1
  have c80681 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.1
  have c80682 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80683 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80684 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_18
  simp only [← c80679, ← c80680, ← c80681] at e_1312_18
  have e_1312_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_19
  simp only [← c80682, ← c80683, k2, ← c80684] at e_1312_19
  have hr := e_1312_19
  simp only [e_1312_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12248)) := by
  have c80685 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80686 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80687 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80688 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80689 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80690 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_0
  simp only [← c80685, ← c80686, ← c80687] at e_1313_0
  have e_1313_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_1
  simp only [← c80688, ← c80689, k2, ← c80690] at e_1313_1
  have hr := e_1313_1
  simp only [e_1313_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12249)) := by
  have c80691 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80692 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80693 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80694 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80695 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80696 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_2
  simp only [← c80691, ← c80692, ← c80693] at e_1313_2
  have e_1313_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_3
  simp only [← c80694, ← c80695, k2, ← c80696] at e_1313_3
  have hr := e_1313_3
  simp only [e_1313_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12250)) := by
  have c80697 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80698 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80699 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80700 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80701 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80702 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_4
  simp only [← c80697, ← c80698, ← c80699] at e_1313_4
  have e_1313_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_5
  simp only [← c80700, ← c80701, k2, ← c80702] at e_1313_5
  have hr := e_1313_5
  simp only [e_1313_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12251)) := by
  have c80703 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80704 := (publicBatchWrapper86_copies2522 a h).1
  have c80705 := (publicBatchWrapper86_copies2522 a h).2.1
  have c80706 := (publicBatchWrapper86_copies2522 a h).2.2.1
  have c80707 := (publicBatchWrapper86_copies2522 a h).2.2.2.1
  have c80708 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_6
  simp only [← c80703, ← c80704, ← c80705] at e_1313_6
  have e_1313_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_7
  simp only [← c80706, ← c80707, k2, ← c80708] at e_1313_7
  have hr := e_1313_7
  simp only [e_1313_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12252)) := by
  have c80709 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.1
  have c80710 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.1
  have c80711 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.1
  have c80712 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.1
  have c80713 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.1
  have c80714 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_8
  simp only [← c80709, ← c80710, ← c80711] at e_1313_8
  have e_1313_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_9
  simp only [← c80712, ← c80713, k2, ← c80714] at e_1313_9
  have hr := e_1313_9
  simp only [e_1313_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12253)) := by
  have c80715 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80716 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80717 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80718 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80719 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80720 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_10
  simp only [← c80715, ← c80716, ← c80717] at e_1313_10
  have e_1313_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_11
  simp only [← c80718, ← c80719, k2, ← c80720] at e_1313_11
  have hr := e_1313_11
  simp only [e_1313_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12254)) := by
  have c80721 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80722 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80723 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80724 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80725 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80726 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_12
  simp only [← c80721, ← c80722, ← c80723] at e_1313_12
  have e_1313_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_13
  simp only [← c80724, ← c80725, k2, ← c80726] at e_1313_13
  have hr := e_1313_13
  simp only [e_1313_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12255)) := by
  have c80727 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80728 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80729 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80730 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80731 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80732 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_14
  simp only [← c80727, ← c80728, ← c80729] at e_1313_14
  have e_1313_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_15
  simp only [← c80730, ← c80731, k2, ← c80732] at e_1313_15
  have hr := e_1313_15
  simp only [e_1313_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12256)) := by
  have c80733 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80734 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80735 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80736 := (publicBatchWrapper86_copies2523 a h).1
  have c80737 := (publicBatchWrapper86_copies2523 a h).2.1
  have c80738 := (publicBatchWrapper86_copies2523 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_16
  simp only [← c80733, ← c80734, ← c80735] at e_1313_16
  have e_1313_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_17
  simp only [← c80736, ← c80737, k2, ← c80738] at e_1313_17
  have hr := e_1313_17
  simp only [e_1313_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12257)) := by
  have c80739 := (publicBatchWrapper86_copies2523 a h).2.2.2.1
  have c80740 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.1
  have c80741 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.1
  have c80742 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.1
  have c80743 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.1
  have c80744 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_18
  simp only [← c80739, ← c80740, ← c80741] at e_1313_18
  have e_1313_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_19
  simp only [← c80742, ← c80743, k2, ← c80744] at e_1313_19
  have hr := e_1313_19
  simp only [e_1313_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12258)) := by
  have c80745 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.1
  have c80746 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80747 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80748 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80749 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80750 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_0
  simp only [← c80745, ← c80746, ← c80747] at e_1314_0
  have e_1314_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_1
  simp only [← c80748, ← c80749, k2, ← c80750] at e_1314_1
  have hr := e_1314_1
  simp only [e_1314_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12259)) := by
  have c80751 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80752 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80753 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80754 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80755 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80756 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_2
  simp only [← c80751, ← c80752, ← c80753] at e_1314_2
  have e_1314_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_3
  simp only [← c80754, ← c80755, k2, ← c80756] at e_1314_3
  have hr := e_1314_3
  simp only [e_1314_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12260)) := by
  have c80757 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80758 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80759 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80760 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80761 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80762 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_4
  simp only [← c80757, ← c80758, ← c80759] at e_1314_4
  have e_1314_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_5
  simp only [← c80760, ← c80761, k2, ← c80762] at e_1314_5
  have hr := e_1314_5
  simp only [e_1314_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12261)) := by
  have c80763 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80764 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80765 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80766 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80767 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80768 := (publicBatchWrapper86_copies2524 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_6
  simp only [← c80763, ← c80764, ← c80765] at e_1314_6
  have e_1314_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_7
  simp only [← c80766, ← c80767, k2, ← c80768] at e_1314_7
  have hr := e_1314_7
  simp only [e_1314_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12262)) := by
  have c80769 := (publicBatchWrapper86_copies2524 a h).2.1
  have c80770 := (publicBatchWrapper86_copies2524 a h).2.2.1
  have c80771 := (publicBatchWrapper86_copies2524 a h).2.2.2.1
  have c80772 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.1
  have c80773 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.1
  have c80774 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_8
  simp only [← c80769, ← c80770, ← c80771] at e_1314_8
  have e_1314_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_9
  simp only [← c80772, ← c80773, k2, ← c80774] at e_1314_9
  have hr := e_1314_9
  simp only [e_1314_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12263)) := by
  have c80775 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.1
  have c80776 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.1
  have c80777 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.1
  have c80778 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80779 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80780 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_10
  simp only [← c80775, ← c80776, ← c80777] at e_1314_10
  have e_1314_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_11
  simp only [← c80778, ← c80779, k2, ← c80780] at e_1314_11
  have hr := e_1314_11
  simp only [e_1314_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12264)) := by
  have c80781 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80782 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80783 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80784 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80785 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80786 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_12
  simp only [← c80781, ← c80782, ← c80783] at e_1314_12
  have e_1314_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_13
  simp only [← c80784, ← c80785, k2, ← c80786] at e_1314_13
  have hr := e_1314_13
  simp only [e_1314_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12265)) := by
  have c80787 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80788 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80789 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80790 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80791 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80792 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_14
  simp only [← c80787, ← c80788, ← c80789] at e_1314_14
  have e_1314_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_15
  simp only [← c80790, ← c80791, k2, ← c80792] at e_1314_15
  have hr := e_1314_15
  simp only [e_1314_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12266)) := by
  have c80793 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80794 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80795 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80796 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80797 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80798 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_16
  simp only [← c80793, ← c80794, ← c80795] at e_1314_16
  have e_1314_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_17
  simp only [← c80796, ← c80797, k2, ← c80798] at e_1314_17
  have hr := e_1314_17
  simp only [e_1314_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12267)) := by
  have c80799 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80800 := (publicBatchWrapper86_copies2525 a h).1
  have c80801 := (publicBatchWrapper86_copies2525 a h).2.1
  have c80802 := (publicBatchWrapper86_copies2525 a h).2.2.1
  have c80803 := (publicBatchWrapper86_copies2525 a h).2.2.2.1
  have c80804 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_18
  simp only [← c80799, ← c80800, ← c80801] at e_1314_18
  have e_1314_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_19
  simp only [← c80802, ← c80803, k2, ← c80804] at e_1314_19
  have hr := e_1314_19
  simp only [e_1314_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12268)) := by
  have c80805 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.1
  have c80806 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.1
  have c80807 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.1
  have c80808 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.1
  have c80809 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.1
  have c80810 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_0
  simp only [← c80805, ← c80806, ← c80807] at e_1315_0
  have e_1315_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_1
  simp only [← c80808, ← c80809, k2, ← c80810] at e_1315_1
  have hr := e_1315_1
  simp only [e_1315_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12269)) := by
  have c80811 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80812 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80813 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80814 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80815 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80816 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_2
  simp only [← c80811, ← c80812, ← c80813] at e_1315_2
  have e_1315_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_3
  simp only [← c80814, ← c80815, k2, ← c80816] at e_1315_3
  have hr := e_1315_3
  simp only [e_1315_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12270)) := by
  have c80817 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80818 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80819 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80820 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80821 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80822 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_4
  simp only [← c80817, ← c80818, ← c80819] at e_1315_4
  have e_1315_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_5
  simp only [← c80820, ← c80821, k2, ← c80822] at e_1315_5
  have hr := e_1315_5
  simp only [e_1315_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12271)) := by
  have c80823 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80824 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80825 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80826 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80827 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80828 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_6
  simp only [← c80823, ← c80824, ← c80825] at e_1315_6
  have e_1315_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_7
  simp only [← c80826, ← c80827, k2, ← c80828] at e_1315_7
  have hr := e_1315_7
  simp only [e_1315_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12272)) := by
  have c80829 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80830 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80831 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80832 := (publicBatchWrapper86_copies2526 a h).1
  have c80833 := (publicBatchWrapper86_copies2526 a h).2.1
  have c80834 := (publicBatchWrapper86_copies2526 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_8
  simp only [← c80829, ← c80830, ← c80831] at e_1315_8
  have e_1315_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_9
  simp only [← c80832, ← c80833, k2, ← c80834] at e_1315_9
  have hr := e_1315_9
  simp only [e_1315_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12273)) := by
  have c80835 := (publicBatchWrapper86_copies2526 a h).2.2.2.1
  have c80836 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.1
  have c80837 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.1
  have c80838 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.1
  have c80839 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.1
  have c80840 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_10
  simp only [← c80835, ← c80836, ← c80837] at e_1315_10
  have e_1315_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_11
  simp only [← c80838, ← c80839, k2, ← c80840] at e_1315_11
  have hr := e_1315_11
  simp only [e_1315_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12274)) := by
  have c80841 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.1
  have c80842 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80843 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80844 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80845 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80846 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_12
  simp only [← c80841, ← c80842, ← c80843] at e_1315_12
  have e_1315_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_13
  simp only [← c80844, ← c80845, k2, ← c80846] at e_1315_13
  have hr := e_1315_13
  simp only [e_1315_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12275)) := by
  have c80847 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80848 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80849 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80850 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80851 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80852 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_14
  simp only [← c80847, ← c80848, ← c80849] at e_1315_14
  have e_1315_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_15
  simp only [← c80850, ← c80851, k2, ← c80852] at e_1315_15
  have hr := e_1315_15
  simp only [e_1315_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12276)) := by
  have c80853 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80854 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80855 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80856 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80857 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80858 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_16
  simp only [← c80853, ← c80854, ← c80855] at e_1315_16
  have e_1315_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_17
  simp only [← c80856, ← c80857, k2, ← c80858] at e_1315_17
  have hr := e_1315_17
  simp only [e_1315_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12277)) := by
  have c80859 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80860 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80861 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80862 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80863 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80864 := (publicBatchWrapper86_copies2527 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_18
  simp only [← c80859, ← c80860, ← c80861] at e_1315_18
  have e_1315_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_19
  simp only [← c80862, ← c80863, k2, ← c80864] at e_1315_19
  have hr := e_1315_19
  simp only [e_1315_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12278)) := by
  have c80865 := (publicBatchWrapper86_copies2527 a h).2.1
  have c80866 := (publicBatchWrapper86_copies2527 a h).2.2.1
  have c80867 := (publicBatchWrapper86_copies2527 a h).2.2.2.1
  have c80868 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.1
  have c80869 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.1
  have c80870 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_0
  simp only [← c80865, ← c80866, ← c80867] at e_1316_0
  have e_1316_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_1
  simp only [← c80868, ← c80869, k2, ← c80870] at e_1316_1
  have hr := e_1316_1
  simp only [e_1316_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12279)) := by
  have c80871 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.1
  have c80872 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.1
  have c80873 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.1
  have c80874 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80875 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80876 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_2
  simp only [← c80871, ← c80872, ← c80873] at e_1316_2
  have e_1316_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_3
  simp only [← c80874, ← c80875, k2, ← c80876] at e_1316_3
  have hr := e_1316_3
  simp only [e_1316_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12280)) := by
  have c80877 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80878 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80879 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80880 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80881 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80882 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_4
  simp only [← c80877, ← c80878, ← c80879] at e_1316_4
  have e_1316_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_5
  simp only [← c80880, ← c80881, k2, ← c80882] at e_1316_5
  have hr := e_1316_5
  simp only [e_1316_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12281)) := by
  have c80883 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80884 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80885 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80886 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80887 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80888 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_6
  simp only [← c80883, ← c80884, ← c80885] at e_1316_6
  have e_1316_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_7
  simp only [← c80886, ← c80887, k2, ← c80888] at e_1316_7
  have hr := e_1316_7
  simp only [e_1316_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12282)) := by
  have c80889 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80890 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80891 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80892 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80893 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80894 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_8
  simp only [← c80889, ← c80890, ← c80891] at e_1316_8
  have e_1316_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_9
  simp only [← c80892, ← c80893, k2, ← c80894] at e_1316_9
  have hr := e_1316_9
  simp only [e_1316_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12283)) := by
  have c80895 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80896 := (publicBatchWrapper86_copies2528 a h).1
  have c80897 := (publicBatchWrapper86_copies2528 a h).2.1
  have c80898 := (publicBatchWrapper86_copies2528 a h).2.2.1
  have c80899 := (publicBatchWrapper86_copies2528 a h).2.2.2.1
  have c80900 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_10
  simp only [← c80895, ← c80896, ← c80897] at e_1316_10
  have e_1316_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_11
  simp only [← c80898, ← c80899, k2, ← c80900] at e_1316_11
  have hr := e_1316_11
  simp only [e_1316_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12284)) := by
  have c80901 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.1
  have c80902 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.1
  have c80903 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.1
  have c80904 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.1
  have c80905 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.1
  have c80906 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_12
  simp only [← c80901, ← c80902, ← c80903] at e_1316_12
  have e_1316_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_13
  simp only [← c80904, ← c80905, k2, ← c80906] at e_1316_13
  have hr := e_1316_13
  simp only [e_1316_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12285)) := by
  have c80907 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80908 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80909 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80910 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80911 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80912 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_14
  simp only [← c80907, ← c80908, ← c80909] at e_1316_14
  have e_1316_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_15
  simp only [← c80910, ← c80911, k2, ← c80912] at e_1316_15
  have hr := e_1316_15
  simp only [e_1316_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12286)) := by
  have c80913 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80914 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80915 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80916 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80917 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80918 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_16
  simp only [← c80913, ← c80914, ← c80915] at e_1316_16
  have e_1316_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_17
  simp only [← c80916, ← c80917, k2, ← c80918] at e_1316_17
  have hr := e_1316_17
  simp only [e_1316_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12287)) := by
  have c80919 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80920 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80921 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80922 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80923 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80924 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_18
  simp only [← c80919, ← c80920, ← c80921] at e_1316_18
  have e_1316_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_19
  simp only [← c80922, ← c80923, k2, ← c80924] at e_1316_19
  have hr := e_1316_19
  simp only [e_1316_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12288)) := by
  have c80925 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80926 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80927 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80928 := (publicBatchWrapper86_copies2529 a h).1
  have c80929 := (publicBatchWrapper86_copies2529 a h).2.1
  have c80930 := (publicBatchWrapper86_copies2529 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_0
  simp only [← c80925, ← c80926, ← c80927] at e_1317_0
  have e_1317_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_1
  simp only [← c80928, ← c80929, k2, ← c80930] at e_1317_1
  have hr := e_1317_1
  simp only [e_1317_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12289)) := by
  have c80931 := (publicBatchWrapper86_copies2529 a h).2.2.2.1
  have c80932 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.1
  have c80933 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.1
  have c80934 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.1
  have c80935 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.1
  have c80936 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_2
  simp only [← c80931, ← c80932, ← c80933] at e_1317_2
  have e_1317_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_3
  simp only [← c80934, ← c80935, k2, ← c80936] at e_1317_3
  have hr := e_1317_3
  simp only [e_1317_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12290)) := by
  have c80937 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.1
  have c80938 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80939 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80940 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80941 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80942 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_4
  simp only [← c80937, ← c80938, ← c80939] at e_1317_4
  have e_1317_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_5
  simp only [← c80940, ← c80941, k2, ← c80942] at e_1317_5
  have hr := e_1317_5
  simp only [e_1317_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12291)) := by
  have c80943 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80944 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80945 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80946 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80947 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80948 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_6
  simp only [← c80943, ← c80944, ← c80945] at e_1317_6
  have e_1317_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_7
  simp only [← c80946, ← c80947, k2, ← c80948] at e_1317_7
  have hr := e_1317_7
  simp only [e_1317_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12292)) := by
  have c80949 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80950 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80951 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80952 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80953 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80954 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_8
  simp only [← c80949, ← c80950, ← c80951] at e_1317_8
  have e_1317_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_9
  simp only [← c80952, ← c80953, k2, ← c80954] at e_1317_9
  have hr := e_1317_9
  simp only [e_1317_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12293)) := by
  have c80955 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80956 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80957 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80958 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80959 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80960 := (publicBatchWrapper86_copies2530 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_10
  simp only [← c80955, ← c80956, ← c80957] at e_1317_10
  have e_1317_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_11
  simp only [← c80958, ← c80959, k2, ← c80960] at e_1317_11
  have hr := e_1317_11
  simp only [e_1317_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12294)) := by
  have c80961 := (publicBatchWrapper86_copies2530 a h).2.1
  have c80962 := (publicBatchWrapper86_copies2530 a h).2.2.1
  have c80963 := (publicBatchWrapper86_copies2530 a h).2.2.2.1
  have c80964 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.1
  have c80965 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.1
  have c80966 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_12
  simp only [← c80961, ← c80962, ← c80963] at e_1317_12
  have e_1317_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_13
  simp only [← c80964, ← c80965, k2, ← c80966] at e_1317_13
  have hr := e_1317_13
  simp only [e_1317_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12295)) := by
  have c80967 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.1
  have c80968 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.1
  have c80969 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.1
  have c80970 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80971 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80972 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_14
  simp only [← c80967, ← c80968, ← c80969] at e_1317_14
  have e_1317_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_15
  simp only [← c80970, ← c80971, k2, ← c80972] at e_1317_15
  have hr := e_1317_15
  simp only [e_1317_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12296)) := by
  have c80973 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80974 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80975 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80976 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80977 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80978 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_16
  simp only [← c80973, ← c80974, ← c80975] at e_1317_16
  have e_1317_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_17
  simp only [← c80976, ← c80977, k2, ← c80978] at e_1317_17
  have hr := e_1317_17
  simp only [e_1317_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12297)) := by
  have c80979 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80980 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80981 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80982 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80983 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80984 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_18
  simp only [← c80979, ← c80980, ← c80981] at e_1317_18
  have e_1317_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_19
  simp only [← c80982, ← c80983, k2, ← c80984] at e_1317_19
  have hr := e_1317_19
  simp only [e_1317_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12298)) := by
  have c80985 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80986 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80987 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80988 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80989 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80990 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_0
  simp only [← c80985, ← c80986, ← c80987] at e_1318_0
  have e_1318_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_1
  simp only [← c80988, ← c80989, k2, ← c80990] at e_1318_1
  have hr := e_1318_1
  simp only [e_1318_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12299)) := by
  have c80991 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80992 := (publicBatchWrapper86_copies2531 a h).1
  have c80993 := (publicBatchWrapper86_copies2531 a h).2.1
  have c80994 := (publicBatchWrapper86_copies2531 a h).2.2.1
  have c80995 := (publicBatchWrapper86_copies2531 a h).2.2.2.1
  have c80996 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_2
  simp only [← c80991, ← c80992, ← c80993] at e_1318_2
  have e_1318_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_3
  simp only [← c80994, ← c80995, k2, ← c80996] at e_1318_3
  have hr := e_1318_3
  simp only [e_1318_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12300)) := by
  have c80997 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.1
  have c80998 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.1
  have c80999 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.1
  have c81000 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.1
  have c81001 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.1
  have c81002 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_4
  simp only [← c80997, ← c80998, ← c80999] at e_1318_4
  have e_1318_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_5
  simp only [← c81000, ← c81001, k2, ← c81002] at e_1318_5
  have hr := e_1318_5
  simp only [e_1318_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12301)) := by
  have c81003 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81004 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81005 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81006 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81007 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81008 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_6
  simp only [← c81003, ← c81004, ← c81005] at e_1318_6
  have e_1318_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_7
  simp only [← c81006, ← c81007, k2, ← c81008] at e_1318_7
  have hr := e_1318_7
  simp only [e_1318_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12302)) := by
  have c81009 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81010 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81011 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81012 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81013 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81014 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_8
  simp only [← c81009, ← c81010, ← c81011] at e_1318_8
  have e_1318_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_9
  simp only [← c81012, ← c81013, k2, ← c81014] at e_1318_9
  have hr := e_1318_9
  simp only [e_1318_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12303)) := by
  have c81015 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81016 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81017 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81018 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81019 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81020 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_10
  simp only [← c81015, ← c81016, ← c81017] at e_1318_10
  have e_1318_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_11
  simp only [← c81018, ← c81019, k2, ← c81020] at e_1318_11
  have hr := e_1318_11
  simp only [e_1318_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12304)) := by
  have c81021 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81022 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81023 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81024 := (publicBatchWrapper86_copies2532 a h).1
  have c81025 := (publicBatchWrapper86_copies2532 a h).2.1
  have c81026 := (publicBatchWrapper86_copies2532 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_12
  simp only [← c81021, ← c81022, ← c81023] at e_1318_12
  have e_1318_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_13
  simp only [← c81024, ← c81025, k2, ← c81026] at e_1318_13
  have hr := e_1318_13
  simp only [e_1318_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12305)) := by
  have c81027 := (publicBatchWrapper86_copies2532 a h).2.2.2.1
  have c81028 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.1
  have c81029 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.1
  have c81030 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.1
  have c81031 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.1
  have c81032 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_14
  simp only [← c81027, ← c81028, ← c81029] at e_1318_14
  have e_1318_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_15
  simp only [← c81030, ← c81031, k2, ← c81032] at e_1318_15
  have hr := e_1318_15
  simp only [e_1318_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12306)) := by
  have c81033 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.1
  have c81034 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81035 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81036 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81037 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81038 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_16
  simp only [← c81033, ← c81034, ← c81035] at e_1318_16
  have e_1318_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_17
  simp only [← c81036, ← c81037, k2, ← c81038] at e_1318_17
  have hr := e_1318_17
  simp only [e_1318_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12307)) := by
  have c81039 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81040 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81041 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81042 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81043 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81044 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_18
  simp only [← c81039, ← c81040, ← c81041] at e_1318_18
  have e_1318_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_19
  simp only [← c81042, ← c81043, k2, ← c81044] at e_1318_19
  have hr := e_1318_19
  simp only [e_1318_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12308)) := by
  have c81045 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81046 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81047 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81048 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81049 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81050 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_0
  simp only [← c81045, ← c81046, ← c81047] at e_1319_0
  have e_1319_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_1
  simp only [← c81048, ← c81049, k2, ← c81050] at e_1319_1
  have hr := e_1319_1
  simp only [e_1319_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12309)) := by
  have c81051 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81052 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81053 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81054 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81055 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81056 := (publicBatchWrapper86_copies2533 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_2
  simp only [← c81051, ← c81052, ← c81053] at e_1319_2
  have e_1319_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_3
  simp only [← c81054, ← c81055, k2, ← c81056] at e_1319_3
  have hr := e_1319_3
  simp only [e_1319_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12310)) := by
  have c81057 := (publicBatchWrapper86_copies2533 a h).2.1
  have c81058 := (publicBatchWrapper86_copies2533 a h).2.2.1
  have c81059 := (publicBatchWrapper86_copies2533 a h).2.2.2.1
  have c81060 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.1
  have c81061 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.1
  have c81062 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_4
  simp only [← c81057, ← c81058, ← c81059] at e_1319_4
  have e_1319_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_5
  simp only [← c81060, ← c81061, k2, ← c81062] at e_1319_5
  have hr := e_1319_5
  simp only [e_1319_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12311)) := by
  have c81063 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.1
  have c81064 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.1
  have c81065 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.1
  have c81066 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81067 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81068 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_6
  simp only [← c81063, ← c81064, ← c81065] at e_1319_6
  have e_1319_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_7
  simp only [← c81066, ← c81067, k2, ← c81068] at e_1319_7
  have hr := e_1319_7
  simp only [e_1319_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12312)) := by
  have c81069 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81070 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81071 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81072 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81073 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81074 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_8
  simp only [← c81069, ← c81070, ← c81071] at e_1319_8
  have e_1319_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_9
  simp only [← c81072, ← c81073, k2, ← c81074] at e_1319_9
  have hr := e_1319_9
  simp only [e_1319_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12313)) := by
  have c81075 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81076 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81077 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81078 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81079 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81080 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_10
  simp only [← c81075, ← c81076, ← c81077] at e_1319_10
  have e_1319_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_11
  simp only [← c81078, ← c81079, k2, ← c81080] at e_1319_11
  have hr := e_1319_11
  simp only [e_1319_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12314)) := by
  have c81081 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81082 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81083 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81084 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81085 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81086 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_12
  simp only [← c81081, ← c81082, ← c81083] at e_1319_12
  have e_1319_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_13
  simp only [← c81084, ← c81085, k2, ← c81086] at e_1319_13
  have hr := e_1319_13
  simp only [e_1319_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12315)) := by
  have c81087 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81088 := (publicBatchWrapper86_copies2534 a h).1
  have c81089 := (publicBatchWrapper86_copies2534 a h).2.1
  have c81090 := (publicBatchWrapper86_copies2534 a h).2.2.1
  have c81091 := (publicBatchWrapper86_copies2534 a h).2.2.2.1
  have c81092 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_14
  simp only [← c81087, ← c81088, ← c81089] at e_1319_14
  have e_1319_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_15
  simp only [← c81090, ← c81091, k2, ← c81092] at e_1319_15
  have hr := e_1319_15
  simp only [e_1319_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12316)) := by
  have c81093 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.1
  have c81094 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.1
  have c81095 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.1
  have c81096 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.1
  have c81097 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.1
  have c81098 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_16
  simp only [← c81093, ← c81094, ← c81095] at e_1319_16
  have e_1319_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_17
  simp only [← c81096, ← c81097, k2, ← c81098] at e_1319_17
  have hr := e_1319_17
  simp only [e_1319_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12317)) := by
  have c81099 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81100 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81101 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81102 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81103 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81104 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_18
  simp only [← c81099, ← c81100, ← c81101] at e_1319_18
  have e_1319_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_19
  simp only [← c81102, ← c81103, k2, ← c81104] at e_1319_19
  have hr := e_1319_19
  simp only [e_1319_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12318)) := by
  have c81105 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81106 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81107 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81108 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81109 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81110 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_0
  simp only [← c81105, ← c81106, ← c81107] at e_1320_0
  have e_1320_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_1
  simp only [← c81108, ← c81109, k2, ← c81110] at e_1320_1
  have hr := e_1320_1
  simp only [e_1320_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12319)) := by
  have c81111 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81112 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81113 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81114 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81115 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81116 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_2
  simp only [← c81111, ← c81112, ← c81113] at e_1320_2
  have e_1320_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_3
  simp only [← c81114, ← c81115, k2, ← c81116] at e_1320_3
  have hr := e_1320_3
  simp only [e_1320_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12320)) := by
  have c81117 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81118 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81119 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81120 := (publicBatchWrapper86_copies2535 a h).1
  have c81121 := (publicBatchWrapper86_copies2535 a h).2.1
  have c81122 := (publicBatchWrapper86_copies2535 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_4
  simp only [← c81117, ← c81118, ← c81119] at e_1320_4
  have e_1320_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_5
  simp only [← c81120, ← c81121, k2, ← c81122] at e_1320_5
  have hr := e_1320_5
  simp only [e_1320_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12321)) := by
  have c81123 := (publicBatchWrapper86_copies2535 a h).2.2.2.1
  have c81124 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.1
  have c81125 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.1
  have c81126 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.1
  have c81127 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.1
  have c81128 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_6
  simp only [← c81123, ← c81124, ← c81125] at e_1320_6
  have e_1320_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_7
  simp only [← c81126, ← c81127, k2, ← c81128] at e_1320_7
  have hr := e_1320_7
  simp only [e_1320_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12322)) := by
  have c81129 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.1
  have c81130 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81131 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81132 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81133 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81134 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_8
  simp only [← c81129, ← c81130, ← c81131] at e_1320_8
  have e_1320_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_9
  simp only [← c81132, ← c81133, k2, ← c81134] at e_1320_9
  have hr := e_1320_9
  simp only [e_1320_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12323)) := by
  have c81135 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81136 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81137 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81138 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81139 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81140 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_10
  simp only [← c81135, ← c81136, ← c81137] at e_1320_10
  have e_1320_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_11
  simp only [← c81138, ← c81139, k2, ← c81140] at e_1320_11
  have hr := e_1320_11
  simp only [e_1320_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12324)) := by
  have c81141 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81142 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81143 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81144 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81145 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81146 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_12
  simp only [← c81141, ← c81142, ← c81143] at e_1320_12
  have e_1320_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_13
  simp only [← c81144, ← c81145, k2, ← c81146] at e_1320_13
  have hr := e_1320_13
  simp only [e_1320_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12325)) := by
  have c81147 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81148 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81149 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81150 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81151 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81152 := (publicBatchWrapper86_copies2536 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_14
  simp only [← c81147, ← c81148, ← c81149] at e_1320_14
  have e_1320_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_15
  simp only [← c81150, ← c81151, k2, ← c81152] at e_1320_15
  have hr := e_1320_15
  simp only [e_1320_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12326)) := by
  have c81153 := (publicBatchWrapper86_copies2536 a h).2.1
  have c81154 := (publicBatchWrapper86_copies2536 a h).2.2.1
  have c81155 := (publicBatchWrapper86_copies2536 a h).2.2.2.1
  have c81156 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.1
  have c81157 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.1
  have c81158 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_16
  simp only [← c81153, ← c81154, ← c81155] at e_1320_16
  have e_1320_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_17
  simp only [← c81156, ← c81157, k2, ← c81158] at e_1320_17
  have hr := e_1320_17
  simp only [e_1320_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12327)) := by
  have c81159 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.1
  have c81160 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.1
  have c81161 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.1
  have c81162 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81163 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81164 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_18
  simp only [← c81159, ← c81160, ← c81161] at e_1320_18
  have e_1320_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_19
  simp only [← c81162, ← c81163, k2, ← c81164] at e_1320_19
  have hr := e_1320_19
  simp only [e_1320_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12328)) := by
  have c81165 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81166 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81167 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81168 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81169 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81170 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_0
  simp only [← c81165, ← c81166, ← c81167] at e_1321_0
  have e_1321_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_1
  simp only [← c81168, ← c81169, k2, ← c81170] at e_1321_1
  have hr := e_1321_1
  simp only [e_1321_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12329)) := by
  have c81171 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81172 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81173 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81174 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81175 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81176 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_2
  simp only [← c81171, ← c81172, ← c81173] at e_1321_2
  have e_1321_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_3
  simp only [← c81174, ← c81175, k2, ← c81176] at e_1321_3
  have hr := e_1321_3
  simp only [e_1321_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12330)) := by
  have c81177 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81178 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81179 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81180 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81181 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81182 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_4
  simp only [← c81177, ← c81178, ← c81179] at e_1321_4
  have e_1321_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_5
  simp only [← c81180, ← c81181, k2, ← c81182] at e_1321_5
  have hr := e_1321_5
  simp only [e_1321_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12331)) := by
  have c81183 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81184 := (publicBatchWrapper86_copies2537 a h).1
  have c81185 := (publicBatchWrapper86_copies2537 a h).2.1
  have c81186 := (publicBatchWrapper86_copies2537 a h).2.2.1
  have c81187 := (publicBatchWrapper86_copies2537 a h).2.2.2.1
  have c81188 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_6
  simp only [← c81183, ← c81184, ← c81185] at e_1321_6
  have e_1321_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_7
  simp only [← c81186, ← c81187, k2, ← c81188] at e_1321_7
  have hr := e_1321_7
  simp only [e_1321_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12332)) := by
  have c81189 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.1
  have c81190 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.1
  have c81191 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.1
  have c81192 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.1
  have c81193 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.1
  have c81194 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_8
  simp only [← c81189, ← c81190, ← c81191] at e_1321_8
  have e_1321_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_9
  simp only [← c81192, ← c81193, k2, ← c81194] at e_1321_9
  have hr := e_1321_9
  simp only [e_1321_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12333)) := by
  have c81195 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81196 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81197 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81198 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81199 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81200 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_10
  simp only [← c81195, ← c81196, ← c81197] at e_1321_10
  have e_1321_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_11
  simp only [← c81198, ← c81199, k2, ← c81200] at e_1321_11
  have hr := e_1321_11
  simp only [e_1321_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12334)) := by
  have c81201 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81202 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81203 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81204 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81205 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81206 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_12
  simp only [← c81201, ← c81202, ← c81203] at e_1321_12
  have e_1321_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_13
  simp only [← c81204, ← c81205, k2, ← c81206] at e_1321_13
  have hr := e_1321_13
  simp only [e_1321_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12335)) := by
  have c81207 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81208 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81209 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81210 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81211 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81212 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_14
  simp only [← c81207, ← c81208, ← c81209] at e_1321_14
  have e_1321_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_15
  simp only [← c81210, ← c81211, k2, ← c81212] at e_1321_15
  have hr := e_1321_15
  simp only [e_1321_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12336)) := by
  have c81213 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81214 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81215 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81216 := (publicBatchWrapper86_copies2538 a h).1
  have c81217 := (publicBatchWrapper86_copies2538 a h).2.1
  have c81218 := (publicBatchWrapper86_copies2538 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_16
  simp only [← c81213, ← c81214, ← c81215] at e_1321_16
  have e_1321_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_17
  simp only [← c81216, ← c81217, k2, ← c81218] at e_1321_17
  have hr := e_1321_17
  simp only [e_1321_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12337)) := by
  have c81219 := (publicBatchWrapper86_copies2538 a h).2.2.2.1
  have c81220 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.1
  have c81221 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.1
  have c81222 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.1
  have c81223 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.1
  have c81224 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_18
  simp only [← c81219, ← c81220, ← c81221] at e_1321_18
  have e_1321_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1321 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_19
  simp only [← c81222, ← c81223, k2, ← c81224] at e_1321_19
  have hr := e_1321_19
  simp only [e_1321_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12338)) := by
  have c81225 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.1
  have c81226 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81227 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81228 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81229 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81230 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_0
  simp only [← c81225, ← c81226, ← c81227] at e_1322_0
  have e_1322_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_1
  simp only [← c81228, ← c81229, k2, ← c81230] at e_1322_1
  have hr := e_1322_1
  simp only [e_1322_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12339)) := by
  have c81231 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81232 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81233 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81234 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81235 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81236 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_2
  simp only [← c81231, ← c81232, ← c81233] at e_1322_2
  have e_1322_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_3
  simp only [← c81234, ← c81235, k2, ← c81236] at e_1322_3
  have hr := e_1322_3
  simp only [e_1322_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12340)) := by
  have c81237 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81238 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81239 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81240 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81241 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81242 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_4
  simp only [← c81237, ← c81238, ← c81239] at e_1322_4
  have e_1322_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_5
  simp only [← c81240, ← c81241, k2, ← c81242] at e_1322_5
  have hr := e_1322_5
  simp only [e_1322_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12341)) := by
  have c81243 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81244 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81245 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81246 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81247 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81248 := (publicBatchWrapper86_copies2539 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_6
  simp only [← c81243, ← c81244, ← c81245] at e_1322_6
  have e_1322_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_7
  simp only [← c81246, ← c81247, k2, ← c81248] at e_1322_7
  have hr := e_1322_7
  simp only [e_1322_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12342)) := by
  have c81249 := (publicBatchWrapper86_copies2539 a h).2.1
  have c81250 := (publicBatchWrapper86_copies2539 a h).2.2.1
  have c81251 := (publicBatchWrapper86_copies2539 a h).2.2.2.1
  have c81252 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.1
  have c81253 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.1
  have c81254 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_8
  simp only [← c81249, ← c81250, ← c81251] at e_1322_8
  have e_1322_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_9
  simp only [← c81252, ← c81253, k2, ← c81254] at e_1322_9
  have hr := e_1322_9
  simp only [e_1322_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12343)) := by
  have c81255 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.1
  have c81256 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.1
  have c81257 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.1
  have c81258 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81259 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81260 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_10
  simp only [← c81255, ← c81256, ← c81257] at e_1322_10
  have e_1322_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_11
  simp only [← c81258, ← c81259, k2, ← c81260] at e_1322_11
  have hr := e_1322_11
  simp only [e_1322_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12344)) := by
  have c81261 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81262 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81263 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81264 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81265 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81266 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_12
  simp only [← c81261, ← c81262, ← c81263] at e_1322_12
  have e_1322_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_13
  simp only [← c81264, ← c81265, k2, ← c81266] at e_1322_13
  have hr := e_1322_13
  simp only [e_1322_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12345)) := by
  have c81267 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81268 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81269 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81270 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81271 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81272 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_14
  simp only [← c81267, ← c81268, ← c81269] at e_1322_14
  have e_1322_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_15
  simp only [← c81270, ← c81271, k2, ← c81272] at e_1322_15
  have hr := e_1322_15
  simp only [e_1322_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12346)) := by
  have c81273 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81274 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81275 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81276 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81277 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81278 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_16
  simp only [← c81273, ← c81274, ← c81275] at e_1322_16
  have e_1322_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_17
  simp only [← c81276, ← c81277, k2, ← c81278] at e_1322_17
  have hr := e_1322_17
  simp only [e_1322_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12347)) := by
  have c81279 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81280 := (publicBatchWrapper86_copies2540 a h).1
  have c81281 := (publicBatchWrapper86_copies2540 a h).2.1
  have c81282 := (publicBatchWrapper86_copies2540 a h).2.2.1
  have c81283 := (publicBatchWrapper86_copies2540 a h).2.2.2.1
  have c81284 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_18
  simp only [← c81279, ← c81280, ← c81281] at e_1322_18
  have e_1322_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1322 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_19
  simp only [← c81282, ← c81283, k2, ← c81284] at e_1322_19
  have hr := e_1322_19
  simp only [e_1322_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12348)) := by
  have c81285 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.1
  have c81286 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.1
  have c81287 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.1
  have c81288 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.1
  have c81289 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.1
  have c81290 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_0
  simp only [← c81285, ← c81286, ← c81287] at e_1323_0
  have e_1323_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_1
  simp only [← c81288, ← c81289, k2, ← c81290] at e_1323_1
  have hr := e_1323_1
  simp only [e_1323_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12349)) := by
  have c81291 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81292 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81293 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81294 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81295 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81296 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_2
  simp only [← c81291, ← c81292, ← c81293] at e_1323_2
  have e_1323_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_3
  simp only [← c81294, ← c81295, k2, ← c81296] at e_1323_3
  have hr := e_1323_3
  simp only [e_1323_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12350)) := by
  have c81297 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81298 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81299 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81300 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81301 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81302 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_4
  simp only [← c81297, ← c81298, ← c81299] at e_1323_4
  have e_1323_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_5
  simp only [← c81300, ← c81301, k2, ← c81302] at e_1323_5
  have hr := e_1323_5
  simp only [e_1323_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12351)) := by
  have c81303 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81304 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81305 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81306 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81307 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81308 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_6
  simp only [← c81303, ← c81304, ← c81305] at e_1323_6
  have e_1323_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_7
  simp only [← c81306, ← c81307, k2, ← c81308] at e_1323_7
  have hr := e_1323_7
  simp only [e_1323_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12352)) := by
  have c81309 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81310 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81311 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81312 := (publicBatchWrapper86_copies2541 a h).1
  have c81313 := (publicBatchWrapper86_copies2541 a h).2.1
  have c81314 := (publicBatchWrapper86_copies2541 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_8
  simp only [← c81309, ← c81310, ← c81311] at e_1323_8
  have e_1323_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_9
  simp only [← c81312, ← c81313, k2, ← c81314] at e_1323_9
  have hr := e_1323_9
  simp only [e_1323_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12353)) := by
  have c81315 := (publicBatchWrapper86_copies2541 a h).2.2.2.1
  have c81316 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.1
  have c81317 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.1
  have c81318 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.1
  have c81319 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.1
  have c81320 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_10
  simp only [← c81315, ← c81316, ← c81317] at e_1323_10
  have e_1323_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_11
  simp only [← c81318, ← c81319, k2, ← c81320] at e_1323_11
  have hr := e_1323_11
  simp only [e_1323_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12354)) := by
  have c81321 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.1
  have c81322 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81323 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81324 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81325 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81326 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_12
  simp only [← c81321, ← c81322, ← c81323] at e_1323_12
  have e_1323_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_13
  simp only [← c81324, ← c81325, k2, ← c81326] at e_1323_13
  have hr := e_1323_13
  simp only [e_1323_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12355)) := by
  have c81327 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81328 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81329 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81330 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81331 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81332 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_14
  simp only [← c81327, ← c81328, ← c81329] at e_1323_14
  have e_1323_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_15
  simp only [← c81330, ← c81331, k2, ← c81332] at e_1323_15
  have hr := e_1323_15
  simp only [e_1323_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12356)) := by
  have c81333 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81334 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81335 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81336 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81337 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81338 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_16
  simp only [← c81333, ← c81334, ← c81335] at e_1323_16
  have e_1323_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_17
  simp only [← c81336, ← c81337, k2, ← c81338] at e_1323_17
  have hr := e_1323_17
  simp only [e_1323_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12357)) := by
  have c81339 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81340 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81341 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81342 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81343 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81344 := (publicBatchWrapper86_copies2542 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_18
  simp only [← c81339, ← c81340, ← c81341] at e_1323_18
  have e_1323_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1323 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_19
  simp only [← c81342, ← c81343, k2, ← c81344] at e_1323_19
  have hr := e_1323_19
  simp only [e_1323_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12358)) := by
  have c81345 := (publicBatchWrapper86_copies2542 a h).2.1
  have c81346 := (publicBatchWrapper86_copies2542 a h).2.2.1
  have c81347 := (publicBatchWrapper86_copies2542 a h).2.2.2.1
  have c81348 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.1
  have c81349 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.1
  have c81350 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_0
  simp only [← c81345, ← c81346, ← c81347] at e_1324_0
  have e_1324_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_1
  simp only [← c81348, ← c81349, k2, ← c81350] at e_1324_1
  have hr := e_1324_1
  simp only [e_1324_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12359)) := by
  have c81351 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.1
  have c81352 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.1
  have c81353 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.1
  have c81354 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81355 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81356 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_2
  simp only [← c81351, ← c81352, ← c81353] at e_1324_2
  have e_1324_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_3
  simp only [← c81354, ← c81355, k2, ← c81356] at e_1324_3
  have hr := e_1324_3
  simp only [e_1324_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12360)) := by
  have c81357 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81358 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81359 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81360 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81361 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81362 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_4
  simp only [← c81357, ← c81358, ← c81359] at e_1324_4
  have e_1324_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_5
  simp only [← c81360, ← c81361, k2, ← c81362] at e_1324_5
  have hr := e_1324_5
  simp only [e_1324_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12361)) := by
  have c81363 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81364 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81365 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81366 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81367 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81368 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_6
  simp only [← c81363, ← c81364, ← c81365] at e_1324_6
  have e_1324_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_7
  simp only [← c81366, ← c81367, k2, ← c81368] at e_1324_7
  have hr := e_1324_7
  simp only [e_1324_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12362)) := by
  have c81369 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81370 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81371 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81372 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81373 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81374 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_8
  simp only [← c81369, ← c81370, ← c81371] at e_1324_8
  have e_1324_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_9
  simp only [← c81372, ← c81373, k2, ← c81374] at e_1324_9
  have hr := e_1324_9
  simp only [e_1324_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12363)) := by
  have c81375 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81376 := (publicBatchWrapper86_copies2543 a h).1
  have c81377 := (publicBatchWrapper86_copies2543 a h).2.1
  have c81378 := (publicBatchWrapper86_copies2543 a h).2.2.1
  have c81379 := (publicBatchWrapper86_copies2543 a h).2.2.2.1
  have c81380 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_10
  simp only [← c81375, ← c81376, ← c81377] at e_1324_10
  have e_1324_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_11
  simp only [← c81378, ← c81379, k2, ← c81380] at e_1324_11
  have hr := e_1324_11
  simp only [e_1324_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12364)) := by
  have c81381 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.1
  have c81382 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.1
  have c81383 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.1
  have c81384 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.1
  have c81385 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.1
  have c81386 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_12
  simp only [← c81381, ← c81382, ← c81383] at e_1324_12
  have e_1324_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_13
  simp only [← c81384, ← c81385, k2, ← c81386] at e_1324_13
  have hr := e_1324_13
  simp only [e_1324_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12365)) := by
  have c81387 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81388 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81389 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81390 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81391 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81392 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_14
  simp only [← c81387, ← c81388, ← c81389] at e_1324_14
  have e_1324_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_15
  simp only [← c81390, ← c81391, k2, ← c81392] at e_1324_15
  have hr := e_1324_15
  simp only [e_1324_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12366)) := by
  have c81393 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81394 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81395 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81396 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81397 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81398 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_16
  simp only [← c81393, ← c81394, ← c81395] at e_1324_16
  have e_1324_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_17
  simp only [← c81396, ← c81397, k2, ← c81398] at e_1324_17
  have hr := e_1324_17
  simp only [e_1324_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12367)) := by
  have c81399 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81400 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81401 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81402 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81403 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81404 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_18
  simp only [← c81399, ← c81400, ← c81401] at e_1324_18
  have e_1324_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1324 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_19
  simp only [← c81402, ← c81403, k2, ← c81404] at e_1324_19
  have hr := e_1324_19
  simp only [e_1324_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12368)) := by
  have c81405 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81406 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81407 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81408 := (publicBatchWrapper86_copies2544 a h).1
  have c81409 := (publicBatchWrapper86_copies2544 a h).2.1
  have c81410 := (publicBatchWrapper86_copies2544 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_0
  simp only [← c81405, ← c81406, ← c81407] at e_1325_0
  have e_1325_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_1
  simp only [← c81408, ← c81409, k2, ← c81410] at e_1325_1
  have hr := e_1325_1
  simp only [e_1325_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12369)) := by
  have c81411 := (publicBatchWrapper86_copies2544 a h).2.2.2.1
  have c81412 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.1
  have c81413 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.1
  have c81414 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.1
  have c81415 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.1
  have c81416 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_2
  simp only [← c81411, ← c81412, ← c81413] at e_1325_2
  have e_1325_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_3
  simp only [← c81414, ← c81415, k2, ← c81416] at e_1325_3
  have hr := e_1325_3
  simp only [e_1325_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12410)) := by
  have c81417 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.1
  have c81418 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81419 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81420 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81421 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81422 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_4
  simp only [← c81417, ← c81418, ← c81419] at e_1325_4
  have e_1325_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_5
  simp only [← c81420, ← c81421, k2, ← c81422] at e_1325_5
  have hr := e_1325_5
  simp only [e_1325_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12411)) := by
  have c81423 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81424 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81425 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81426 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81427 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81428 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_6
  simp only [← c81423, ← c81424, ← c81425] at e_1325_6
  have e_1325_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_7
  simp only [← c81426, ← c81427, k2, ← c81428] at e_1325_7
  have hr := e_1325_7
  simp only [e_1325_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12412)) := by
  have c81429 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81430 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81431 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81432 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81433 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81434 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_8
  simp only [← c81429, ← c81430, ← c81431] at e_1325_8
  have e_1325_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_9
  simp only [← c81432, ← c81433, k2, ← c81434] at e_1325_9
  have hr := e_1325_9
  simp only [e_1325_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12413)) := by
  have c81435 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81436 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81437 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81438 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81439 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81440 := (publicBatchWrapper86_copies2545 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_10
  simp only [← c81435, ← c81436, ← c81437] at e_1325_10
  have e_1325_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_11
  simp only [← c81438, ← c81439, k2, ← c81440] at e_1325_11
  have hr := e_1325_11
  simp only [e_1325_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12414)) := by
  have c81441 := (publicBatchWrapper86_copies2545 a h).2.1
  have c81442 := (publicBatchWrapper86_copies2545 a h).2.2.1
  have c81443 := (publicBatchWrapper86_copies2545 a h).2.2.2.1
  have c81444 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.1
  have c81445 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.1
  have c81446 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_12
  simp only [← c81441, ← c81442, ← c81443] at e_1325_12
  have e_1325_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_13
  simp only [← c81444, ← c81445, k2, ← c81446] at e_1325_13
  have hr := e_1325_13
  simp only [e_1325_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12415)) := by
  have c81447 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.1
  have c81448 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.1
  have c81449 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.1
  have c81450 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81451 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81452 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_14
  simp only [← c81447, ← c81448, ← c81449] at e_1325_14
  have e_1325_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_15
  simp only [← c81450, ← c81451, k2, ← c81452] at e_1325_15
  have hr := e_1325_15
  simp only [e_1325_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12416)) := by
  have c81453 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81454 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81455 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81456 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81457 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81458 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_16
  simp only [← c81453, ← c81454, ← c81455] at e_1325_16
  have e_1325_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_17
  simp only [← c81456, ← c81457, k2, ← c81458] at e_1325_17
  have hr := e_1325_17
  simp only [e_1325_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12417)) := by
  have c81459 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81460 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81461 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81462 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81463 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81464 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_18
  simp only [← c81459, ← c81460, ← c81461] at e_1325_18
  have e_1325_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1325 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_19
  simp only [← c81462, ← c81463, k2, ← c81464] at e_1325_19
  have hr := e_1325_19
  simp only [e_1325_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12418)) := by
  have c81465 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81466 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81467 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81468 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81469 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81470 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_0
  simp only [← c81465, ← c81466, ← c81467] at e_1326_0
  have e_1326_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_1
  simp only [← c81468, ← c81469, k2, ← c81470] at e_1326_1
  have hr := e_1326_1
  simp only [e_1326_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12419)) := by
  have c81471 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81472 := (publicBatchWrapper86_copies2546 a h).1
  have c81473 := (publicBatchWrapper86_copies2546 a h).2.1
  have c81474 := (publicBatchWrapper86_copies2546 a h).2.2.1
  have c81475 := (publicBatchWrapper86_copies2546 a h).2.2.2.1
  have c81476 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_2
  simp only [← c81471, ← c81472, ← c81473] at e_1326_2
  have e_1326_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_3
  simp only [← c81474, ← c81475, k2, ← c81476] at e_1326_3
  have hr := e_1326_3
  simp only [e_1326_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12420)) := by
  have c81477 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.1
  have c81478 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.1
  have c81479 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.1
  have c81480 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.1
  have c81481 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.1
  have c81482 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_4
  simp only [← c81477, ← c81478, ← c81479] at e_1326_4
  have e_1326_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_5
  simp only [← c81480, ← c81481, k2, ← c81482] at e_1326_5
  have hr := e_1326_5
  simp only [e_1326_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12421)) := by
  have c81483 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81484 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81485 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81486 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81487 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81488 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_6
  simp only [← c81483, ← c81484, ← c81485] at e_1326_6
  have e_1326_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_7
  simp only [← c81486, ← c81487, k2, ← c81488] at e_1326_7
  have hr := e_1326_7
  simp only [e_1326_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12422)) := by
  have c81489 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81490 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81491 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81492 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81493 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81494 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_8
  simp only [← c81489, ← c81490, ← c81491] at e_1326_8
  have e_1326_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_9
  simp only [← c81492, ← c81493, k2, ← c81494] at e_1326_9
  have hr := e_1326_9
  simp only [e_1326_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12423)) := by
  have c81495 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81496 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81497 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81498 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81499 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81500 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_10
  simp only [← c81495, ← c81496, ← c81497] at e_1326_10
  have e_1326_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_11
  simp only [← c81498, ← c81499, k2, ← c81500] at e_1326_11
  have hr := e_1326_11
  simp only [e_1326_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12424)) := by
  have c81501 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81502 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81503 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81504 := (publicBatchWrapper86_copies2547 a h).1
  have c81505 := (publicBatchWrapper86_copies2547 a h).2.1
  have c81506 := (publicBatchWrapper86_copies2547 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_12
  simp only [← c81501, ← c81502, ← c81503] at e_1326_12
  have e_1326_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_13
  simp only [← c81504, ← c81505, k2, ← c81506] at e_1326_13
  have hr := e_1326_13
  simp only [e_1326_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12425)) := by
  have c81507 := (publicBatchWrapper86_copies2547 a h).2.2.2.1
  have c81508 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.1
  have c81509 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.1
  have c81510 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.1
  have c81511 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.1
  have c81512 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_14
  simp only [← c81507, ← c81508, ← c81509] at e_1326_14
  have e_1326_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_15
  simp only [← c81510, ← c81511, k2, ← c81512] at e_1326_15
  have hr := e_1326_15
  simp only [e_1326_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12426)) := by
  have c81513 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.1
  have c81514 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81515 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81516 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81517 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81518 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_16
  simp only [← c81513, ← c81514, ← c81515] at e_1326_16
  have e_1326_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_17
  simp only [← c81516, ← c81517, k2, ← c81518] at e_1326_17
  have hr := e_1326_17
  simp only [e_1326_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12427)) := by
  have c81519 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81520 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81521 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81522 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81523 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81524 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_18
  simp only [← c81519, ← c81520, ← c81521] at e_1326_18
  have e_1326_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1326 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_19
  simp only [← c81522, ← c81523, k2, ← c81524] at e_1326_19
  have hr := e_1326_19
  simp only [e_1326_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12428)) := by
  have c81525 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81526 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81527 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81528 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81529 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81530 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_0
  simp only [← c81525, ← c81526, ← c81527] at e_1327_0
  have e_1327_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_1
  simp only [← c81528, ← c81529, k2, ← c81530] at e_1327_1
  have hr := e_1327_1
  simp only [e_1327_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12429)) := by
  have c81531 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81532 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81533 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81534 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81535 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81536 := (publicBatchWrapper86_copies2548 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_2
  simp only [← c81531, ← c81532, ← c81533] at e_1327_2
  have e_1327_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_3
  simp only [← c81534, ← c81535, k2, ← c81536] at e_1327_3
  have hr := e_1327_3
  simp only [e_1327_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12430)) := by
  have c81537 := (publicBatchWrapper86_copies2548 a h).2.1
  have c81538 := (publicBatchWrapper86_copies2548 a h).2.2.1
  have c81539 := (publicBatchWrapper86_copies2548 a h).2.2.2.1
  have c81540 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.1
  have c81541 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.1
  have c81542 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_4
  simp only [← c81537, ← c81538, ← c81539] at e_1327_4
  have e_1327_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_5
  simp only [← c81540, ← c81541, k2, ← c81542] at e_1327_5
  have hr := e_1327_5
  simp only [e_1327_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12431)) := by
  have c81543 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.1
  have c81544 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.1
  have c81545 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.1
  have c81546 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81547 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81548 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_6
  simp only [← c81543, ← c81544, ← c81545] at e_1327_6
  have e_1327_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_7
  simp only [← c81546, ← c81547, k2, ← c81548] at e_1327_7
  have hr := e_1327_7
  simp only [e_1327_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12432)) := by
  have c81549 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81550 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81551 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81552 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81553 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81554 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_8
  simp only [← c81549, ← c81550, ← c81551] at e_1327_8
  have e_1327_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_9
  simp only [← c81552, ← c81553, k2, ← c81554] at e_1327_9
  have hr := e_1327_9
  simp only [e_1327_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12433)) := by
  have c81555 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81556 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81557 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81558 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81559 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81560 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_10
  simp only [← c81555, ← c81556, ← c81557] at e_1327_10
  have e_1327_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_11
  simp only [← c81558, ← c81559, k2, ← c81560] at e_1327_11
  have hr := e_1327_11
  simp only [e_1327_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12434)) := by
  have c81561 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81562 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81563 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81564 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81565 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81566 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_12
  simp only [← c81561, ← c81562, ← c81563] at e_1327_12
  have e_1327_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_13
  simp only [← c81564, ← c81565, k2, ← c81566] at e_1327_13
  have hr := e_1327_13
  simp only [e_1327_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12435)) := by
  have c81567 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81568 := (publicBatchWrapper86_copies2549 a h).1
  have c81569 := (publicBatchWrapper86_copies2549 a h).2.1
  have c81570 := (publicBatchWrapper86_copies2549 a h).2.2.1
  have c81571 := (publicBatchWrapper86_copies2549 a h).2.2.2.1
  have c81572 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_14
  simp only [← c81567, ← c81568, ← c81569] at e_1327_14
  have e_1327_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_15
  simp only [← c81570, ← c81571, k2, ← c81572] at e_1327_15
  have hr := e_1327_15
  simp only [e_1327_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12436)) := by
  have c81573 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.1
  have c81574 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.1
  have c81575 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.1
  have c81576 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.1
  have c81577 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.1
  have c81578 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_16
  simp only [← c81573, ← c81574, ← c81575] at e_1327_16
  have e_1327_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_17
  simp only [← c81576, ← c81577, k2, ← c81578] at e_1327_17
  have hr := e_1327_17
  simp only [e_1327_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12437)) := by
  have c81579 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81580 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81581 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81582 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81583 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81584 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_18
  simp only [← c81579, ← c81580, ← c81581] at e_1327_18
  have e_1327_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1327 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_19
  simp only [← c81582, ← c81583, k2, ← c81584] at e_1327_19
  have hr := e_1327_19
  simp only [e_1327_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12438)) := by
  have c81585 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81586 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81587 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81588 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81589 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81590 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_0
  simp only [← c81585, ← c81586, ← c81587] at e_1328_0
  have e_1328_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_1
  simp only [← c81588, ← c81589, k2, ← c81590] at e_1328_1
  have hr := e_1328_1
  simp only [e_1328_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12439)) := by
  have c81591 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81592 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81593 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81594 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81595 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81596 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_2
  simp only [← c81591, ← c81592, ← c81593] at e_1328_2
  have e_1328_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_3
  simp only [← c81594, ← c81595, k2, ← c81596] at e_1328_3
  have hr := e_1328_3
  simp only [e_1328_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12440)) := by
  have c81597 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81598 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81599 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81600 := (publicBatchWrapper86_copies2550 a h).1
  have c81601 := (publicBatchWrapper86_copies2550 a h).2.1
  have c81602 := (publicBatchWrapper86_copies2550 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_4
  simp only [← c81597, ← c81598, ← c81599] at e_1328_4
  have e_1328_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_5
  simp only [← c81600, ← c81601, k2, ← c81602] at e_1328_5
  have hr := e_1328_5
  simp only [e_1328_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12441)) := by
  have c81603 := (publicBatchWrapper86_copies2550 a h).2.2.2.1
  have c81604 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.1
  have c81605 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.1
  have c81606 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.1
  have c81607 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.1
  have c81608 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_6
  simp only [← c81603, ← c81604, ← c81605] at e_1328_6
  have e_1328_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_7
  simp only [← c81606, ← c81607, k2, ← c81608] at e_1328_7
  have hr := e_1328_7
  simp only [e_1328_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12442)) := by
  have c81609 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.1
  have c81610 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81611 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81612 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81613 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81614 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_8
  simp only [← c81609, ← c81610, ← c81611] at e_1328_8
  have e_1328_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_9
  simp only [← c81612, ← c81613, k2, ← c81614] at e_1328_9
  have hr := e_1328_9
  simp only [e_1328_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
