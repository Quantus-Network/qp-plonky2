/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PublicBatchWrapper89`, split out of the recorded wrapper
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.Generated.PublicBatchWrapper89.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option maxRecDepth 200000

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

theorem publicBatchWrapper89_f20101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17789)) := by
  have c125112 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125113 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125114 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125115 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125116 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125117 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_0 := arithEq_of_rows h (row := 2052) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_0
  simp only [← c125112, ← c125113, ← c125114] at e_2052_0
  have e_2052_1 := arithEq_of_rows h (row := 2052) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_1
  simp only [← c125115, ← c125116, k2, ← c125117] at e_2052_1
  have hr := e_2052_1
  simp only [e_2052_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17790)) := by
  have c125118 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125119 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125120 := (publicBatchWrapper89_copies3910 a h).1
  have c125121 := (publicBatchWrapper89_copies3910 a h).2.1
  have c125122 := (publicBatchWrapper89_copies3910 a h).2.2.1
  have c125123 := (publicBatchWrapper89_copies3910 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_2 := arithEq_of_rows h (row := 2052) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_2
  simp only [← c125118, ← c125119, ← c125120] at e_2052_2
  have e_2052_3 := arithEq_of_rows h (row := 2052) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_3
  simp only [← c125121, ← c125122, k2, ← c125123] at e_2052_3
  have hr := e_2052_3
  simp only [e_2052_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17791)) := by
  have c125124 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.1
  have c125125 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.1
  have c125126 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.1
  have c125127 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.1
  have c125128 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.1
  have c125129 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_4 := arithEq_of_rows h (row := 2052) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_4
  simp only [← c125124, ← c125125, ← c125126] at e_2052_4
  have e_2052_5 := arithEq_of_rows h (row := 2052) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_5
  simp only [← c125127, ← c125128, k2, ← c125129] at e_2052_5
  have hr := e_2052_5
  simp only [e_2052_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17792)) := by
  have c125130 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125131 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125132 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125133 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125134 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125135 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_6 := arithEq_of_rows h (row := 2052) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_6
  simp only [← c125130, ← c125131, ← c125132] at e_2052_6
  have e_2052_7 := arithEq_of_rows h (row := 2052) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_7
  simp only [← c125133, ← c125134, k2, ← c125135] at e_2052_7
  have hr := e_2052_7
  simp only [e_2052_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17793)) := by
  have c125136 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125137 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125138 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125139 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125140 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125141 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_8 := arithEq_of_rows h (row := 2052) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_8
  simp only [← c125136, ← c125137, ← c125138] at e_2052_8
  have e_2052_9 := arithEq_of_rows h (row := 2052) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_9
  simp only [← c125139, ← c125140, k2, ← c125141] at e_2052_9
  have hr := e_2052_9
  simp only [e_2052_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17794)) := by
  have c125142 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125143 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125144 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125145 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125146 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125147 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_10 := arithEq_of_rows h (row := 2052) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_10
  simp only [← c125142, ← c125143, ← c125144] at e_2052_10
  have e_2052_11 := arithEq_of_rows h (row := 2052) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_11
  simp only [← c125145, ← c125146, k2, ← c125147] at e_2052_11
  have hr := e_2052_11
  simp only [e_2052_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17795)) := by
  have c125148 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125149 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125150 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125151 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125152 := (publicBatchWrapper89_copies3911 a h).1
  have c125153 := (publicBatchWrapper89_copies3911 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_12 := arithEq_of_rows h (row := 2052) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_12
  simp only [← c125148, ← c125149, ← c125150] at e_2052_12
  have e_2052_13 := arithEq_of_rows h (row := 2052) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_13
  simp only [← c125151, ← c125152, k2, ← c125153] at e_2052_13
  have hr := e_2052_13
  simp only [e_2052_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17796)) := by
  have c125154 := (publicBatchWrapper89_copies3911 a h).2.2.1
  have c125155 := (publicBatchWrapper89_copies3911 a h).2.2.2.1
  have c125156 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.1
  have c125157 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.1
  have c125158 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.1
  have c125159 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_14 := arithEq_of_rows h (row := 2052) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_14
  simp only [← c125154, ← c125155, ← c125156] at e_2052_14
  have e_2052_15 := arithEq_of_rows h (row := 2052) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_15
  simp only [← c125157, ← c125158, k2, ← c125159] at e_2052_15
  have hr := e_2052_15
  simp only [e_2052_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17797)) := by
  have c125160 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.1
  have c125161 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.1
  have c125162 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125163 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125164 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125165 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_16 := arithEq_of_rows h (row := 2052) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_16
  simp only [← c125160, ← c125161, ← c125162] at e_2052_16
  have e_2052_17 := arithEq_of_rows h (row := 2052) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_17
  simp only [← c125163, ← c125164, k2, ← c125165] at e_2052_17
  have hr := e_2052_17
  simp only [e_2052_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17798)) := by
  have c125166 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125167 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125168 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125169 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125170 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125171 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_18 := arithEq_of_rows h (row := 2052) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_18
  simp only [← c125166, ← c125167, ← c125168] at e_2052_18
  have e_2052_19 := arithEq_of_rows h (row := 2052) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_19
  simp only [← c125169, ← c125170, k2, ← c125171] at e_2052_19
  have hr := e_2052_19
  simp only [e_2052_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2053 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17799)) := by
  have c125172 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125173 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125174 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125175 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125176 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125177 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2053_0 := arithEq_of_rows h (row := 2053) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_0
  simp only [← c125172, ← c125173, ← c125174] at e_2053_0
  have e_2053_1 := arithEq_of_rows h (row := 2053) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_1
  simp only [← c125175, ← c125176, k2, ← c125177] at e_2053_1
  have hr := e_2053_1
  simp only [e_2053_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2053 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17800)) := by
  have c125178 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125179 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125180 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125181 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125182 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125183 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2053_2 := arithEq_of_rows h (row := 2053) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_2
  simp only [← c125178, ← c125179, ← c125180] at e_2053_2
  have e_2053_3 := arithEq_of_rows h (row := 2053) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_3
  simp only [← c125181, ← c125182, k2, ← c125183] at e_2053_3
  have hr := e_2053_3
  simp only [e_2053_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2053 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17801)) := by
  have c125184 := (publicBatchWrapper89_copies3912 a h).1
  have c125185 := (publicBatchWrapper89_copies3912 a h).2.1
  have c125186 := (publicBatchWrapper89_copies3912 a h).2.2.1
  have c125187 := (publicBatchWrapper89_copies3912 a h).2.2.2.1
  have c125188 := (publicBatchWrapper89_copies3912 a h).2.2.2.2.1
  have c125189 := (publicBatchWrapper89_copies3912 a h).2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2053_4 := arithEq_of_rows h (row := 2053) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_4
  simp only [← c125184, ← c125185, ← c125186] at e_2053_4
  have e_2053_5 := arithEq_of_rows h (row := 2053) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_5
  simp only [← c125187, ← c125188, k2, ← c125189] at e_2053_5
  have hr := e_2053_5
  simp only [e_2053_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
